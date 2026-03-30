#!/usr/bin/env python3
"""Run RF-DETR inference on a video and optionally save overlayed output.

Supports --backend pytorch (default) or --backend onnx with --onnx-model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import torch

from infer_helpers import (
    detect_model_size_from_checkpoint,
    instantiate_model,
    read_checkpoint_args,
)
from infer_webcam import (
    draw_detections,
    load_class_names,
    load_onnx_session,
    onnx_input_hw,
    run_onnx_frame,
)
from moldvision.checkpoints import load_checkpoint_weights
from moldvision.postprocess import letterbox_pil, parse_model_output_generic, unletterbox_mask, unletterbox_xyxy
from moldvision.torch_compat import unwrap_torch_module
import torchvision.transforms as T
from PIL import Image
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="ARIA_MoldVision: Run inference on a video file")
    p.add_argument("--video", "-i", required=True, help="input video path")
    p.add_argument("--weights", "-w", default=None, help="path to model weights (.pth); required when --backend pytorch")
    p.add_argument("--task", choices=["detect", "seg"], default="detect", help="inference task")
    p.add_argument(
        "--size",
        choices=["nano", "small", "base", "medium"],
        default="nano",
        help="model size to instantiate; for seg this is auto-detected from checkpoint when possible",
    )
    p.add_argument("--num-classes", type=int, default=None, help="override number of classes for detection models")
    p.add_argument("--threshold", type=float, default=0.3, help="score threshold for displayed detections")
    p.add_argument("--mask-alpha", type=float, default=0.45, help="mask overlay alpha in [0..1], seg only")
    p.add_argument("--mask-thresh", type=float, default=0.5, help="threshold for mask probabilities, seg only")
    p.add_argument("--classes-file", type=str, default=None, help="path to METADATA.json or newline class file")
    p.add_argument("--device", type=str, default=None, help="torch device to use (default: auto-detect)")
    p.add_argument("--checkpoint-key", type=str, default=None, help="explicit key inside checkpoint for state_dict")
    p.add_argument("--use-checkpoint-model", action="store_true", help="use a pickled model object from checkpoint if present")
    p.add_argument("--verbose", action="store_true", help="print additional loading/progress information")
    p.add_argument("--display", action="store_true", help="display the overlayed video while processing")
    p.add_argument("--width", type=int, default=None, help="optional display width")
    p.add_argument("--max-frames", type=int, default=None, help="optional frame limit for quick smoke runs")
    p.add_argument("--frame-step", type=int, default=1, help="process every Nth frame (default: 1)")
    p.add_argument("--codec", default="mp4v", help="fourcc codec for saved video, default: mp4v")
    p.add_argument("--output-fps", type=float, default=None, help="override output video fps")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--output-video", "-o", default=None, help="path to save the overlayed video")
    out.add_argument(
        "--out-dir",
        default=None,
        help="directory where the overlayed video will be saved with an auto-generated filename",
    )
    # ONNX backend
    p.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch",
                   help="inference backend: pytorch (default) or onnx")
    p.add_argument("--onnx-model", type=str, default=None,
                   help="path to ONNX model file (required when --backend onnx)")
    return p.parse_args()


def resolve_output_path(args) -> Path | None:
    if args.output_video:
        return Path(args.output_video)
    if args.out_dir:
        video_path = Path(args.video)
        suffix = video_path.suffix if video_path.suffix else ".mp4"
        return Path(args.out_dir) / f"{video_path.stem}_overlay{suffix}"
    return None


def preprocess_frame_to_tensor(frame, device: torch.device, *, target_w: int, target_h: int):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil_in, lb = letterbox_pil(pil, target_w=int(target_w), target_h=int(target_h))
    tensor = T.ToTensor()(pil_in).unsqueeze(0).to(device)
    return tensor, lb


def run_official_predict(model, frame_bgr, *, threshold: float):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    predict_fn = getattr(model, "predict", None)
    if not callable(predict_fn):
        return None
    with torch.inference_mode():
        try:
            return predict_fn(pil, threshold=float(threshold))
        except TypeError:
            return predict_fn(pil)


def _mask_color(label_id: int) -> tuple[int, int, int]:
    palette = [
        (0, 255, 0),
        (0, 200, 255),
        (255, 180, 0),
        (255, 80, 80),
        (255, 0, 255),
        (0, 255, 255),
    ]
    return palette[int(label_id) % len(palette)]


def overlay_masks_visible(frame_bgr: np.ndarray, masks, labels, alpha: float = 0.45) -> None:
    if not masks:
        return
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    for i, m in enumerate(masks):
        if m is None:
            continue
        mm = np.asarray(m).astype(bool)
        if mm.ndim != 2:
            continue
        if mm.shape[0] != h or mm.shape[1] != w:
            mm_u8 = (mm.astype(np.uint8) * 255)
            mm_u8 = cv2.resize(mm_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            mm = mm_u8 > 127
        color = _mask_color(labels[i] if i < len(labels) else i)
        overlay[mm] = color
        contours, _ = cv2.findContours((mm.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_bgr, contours, -1, color, 2)
    cv2.addWeighted(overlay, float(alpha), frame_bgr, float(1.0 - alpha), 0, dst=frame_bgr)


def create_writer(out_path: Path, codec: str, fps: float, frame_width: int, frame_height: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec[:4])
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(frame_width), int(frame_height)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {out_path}")
    return writer


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    class_names = load_class_names(args.classes_file) if args.classes_file else []

    if args.backend == "onnx":
        if not args.onnx_model:
            print("Error: --onnx-model is required when --backend onnx")
            return
        session = load_onnx_session(args.onnx_model, args.device)
        onnx_target_h, onnx_target_w = onnx_input_hw(session)
        model = None
        module_for_torch = None
        infer_resolution = onnx_target_w
    else:
        if not args.weights:
            print("Error: --weights is required when --backend pytorch")
            return
        ckpt_args = read_checkpoint_args(args.weights)
        if not class_names and ckpt_args is not None:
            class_names = list(getattr(ckpt_args, "class_names", []) or [])

        model_size = args.size
        detected_size = detect_model_size_from_checkpoint(args.weights, checkpoint_key=args.checkpoint_key)
        if args.task == "seg" and detected_size:
            model_size = detected_size
            print(f"Using segmentation model size from checkpoint: {model_size}")

        model_num_classes = args.num_classes
        if model_num_classes is None and class_names:
            model_num_classes = int(len(class_names))
        if model_num_classes is None and ckpt_args is not None:
            ck_num_classes = getattr(ckpt_args, "num_classes", None)
            if isinstance(ck_num_classes, int) and ck_num_classes > 0:
                model_num_classes = int(ck_num_classes)

        infer_resolution = 640
        if ckpt_args is not None:
            ck_resolution = getattr(ckpt_args, "resolution", None)
            if isinstance(ck_resolution, int) and ck_resolution > 0:
                infer_resolution = int(ck_resolution)
        if args.verbose:
            print(f"Inference config: size={model_size} num_classes={model_num_classes} resolution={infer_resolution}")

        model = instantiate_model(size=model_size, num_classes=model_num_classes, task=args.task)
        lr = load_checkpoint_weights(
            model,
            args.weights,
            device,
            checkpoint_key=args.checkpoint_key,
            allow_replace_model=args.use_checkpoint_model,
            strict=False,
            verbose=args.verbose,
        )
        if lr.replacement_model is not None:
            model = lr.replacement_model
        if not lr.ok:
            print(f"Warning: could not load weights cleanly: {lr.message}")

        try:
            model.to(device)
        except Exception:
            pass
        try:
            model.eval()
        except Exception:
            pass
        optimize = getattr(model, "optimize_for_inference", None)
        if callable(optimize):
            try:
                optimize(compile=False, batch_size=1)
            except Exception:
                pass
        try:
            module_for_torch = unwrap_torch_module(model)
        except Exception:
            module_for_torch = model if hasattr(model, "__call__") else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    out_path = resolve_output_path(args)
    writer = None
    processed = 0
    warned_no_masks = False

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        if args.output_fps is not None and args.output_fps > 0:
            fps = float(args.output_fps)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if args.frame_step > 1 and processed % args.frame_step != 0:
                processed += 1
                continue

            h, w = frame.shape[:2]

            if args.backend == "onnx":
                boxes, scores, labels, masks = run_onnx_frame(
                    session, frame, onnx_target_h, onnx_target_w,
                    args.threshold, args.task == "seg", args.mask_thresh,
                )
            else:
                out = None
                last_exc = None

                try:
                    out = run_official_predict(model, frame, threshold=float(args.threshold))
                except Exception as e:
                    last_exc = e

                if out is not None:
                    boxes, scores, labels, masks = parse_model_output_generic(
                        out,
                        img_w=w,
                        img_h=h,
                        score_thresh=args.threshold,
                        want_masks=(args.task == "seg"),
                        mask_thresh=args.mask_thresh,
                        topk=300,
                    )
                else:
                    tensor, lb = preprocess_frame_to_tensor(
                        frame,
                        device,
                        target_w=int(infer_resolution),
                        target_h=int(infer_resolution),
                    )
                    if module_for_torch is not None:
                        try:
                            with torch.inference_mode():
                                out = module_for_torch(tensor)
                        except Exception as e:
                            last_exc = e
                    if out is None:
                        for name in ("infer", "inference", "forward", "detect"):
                            fn = getattr(model, name, None)
                            if not callable(fn):
                                continue
                            try:
                                with torch.inference_mode():
                                    out = fn(tensor)
                                break
                            except Exception as e:
                                last_exc = e
                    if out is None:
                        raise RuntimeError(f"Inference failed: {last_exc}")

                    boxes, scores, labels, masks = parse_model_output_generic(
                        out,
                        img_w=int(infer_resolution),
                        img_h=int(infer_resolution),
                        score_thresh=args.threshold,
                        want_masks=(args.task == "seg"),
                        mask_thresh=args.mask_thresh,
                        topk=300,
                    )
                    boxes = [unletterbox_xyxy(b, lb=lb, orig_w=w, orig_h=h) for b in boxes]
                    if masks is not None:
                        masks = [unletterbox_mask(m, lb=lb, orig_w=w, orig_h=h) for m in masks]
                        masks = [m for m in masks if m is not None]

            if args.task == "seg" and boxes and not masks and not warned_no_masks:
                print(
                    "Warning: segmentation inference returned boxes but no masks. "
                    "Official predict() was tried first, then raw-output fallback."
                )
                warned_no_masks = True

            disp = frame.copy()
            if args.task == "seg" and masks:
                overlay_masks_visible(disp, masks, labels, alpha=args.mask_alpha)
            draw_detections(disp, boxes, scores, labels, class_names)

            if writer is None and out_path is not None:
                writer = create_writer(out_path, args.codec, fps, disp.shape[1], disp.shape[0])

            if writer is not None:
                writer.write(disp)

            if args.display:
                shown = disp
                if args.width and args.width > 0 and shown.shape[1] != args.width:
                    scale = float(args.width) / float(shown.shape[1])
                    shown = cv2.resize(shown, (args.width, int(shown.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                cv2.imshow("ARIA_MoldVision Video Inference (press q to quit)", shown)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            processed += 1
            if args.verbose and processed % 25 == 0:
                if total_frames > 0:
                    print(f"Processed {processed}/{total_frames} frames")
                else:
                    print(f"Processed {processed} frames")
            if args.max_frames is not None and processed >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if out_path is not None:
        print(f"Wrote overlay video to: {out_path}")
    print(f"Processed frames: {processed}")


if __name__ == "__main__":
    main()
