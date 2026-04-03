#!/usr/bin/env python3
"""Run inference on a single image and overlay boxes + instance masks.

Usage example:
  python scripts/infer_image.py --image data/val/0001.jpg --weights datasets/<UUID>/models/checkpoint_best_regular.pth --task seg --output out.png --classes-file datasets/<UUID>/METADATA.json
  python scripts/infer_image.py --backend onnx --onnx-model model.onnx --image data/val/0001.jpg --output out.png

The script is flexible: pick `--task seg` or `detect`, `--size` (for detect), `--threshold`, `--mask-thresh`, and device.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from infer_helpers import (
    detect_model_size_from_checkpoint,
    instantiate_model,
    load_checkpoint_weights,
    parse_model_output,
    detections_to_json,
    read_checkpoint_args,
)
from infer_webcam import run_inference, load_onnx_session, onnx_input_hw, run_onnx_frame
from moldvision.infer import InferenceEngine


def _call_predict_best_effort(predict_fn, image: Image.Image, *, threshold: float):
    """Call a wrapper predict() with a best-effort threshold kwarg if supported."""
    try:
        import inspect as _inspect

        sig = _inspect.signature(predict_fn)
        params = sig.parameters
        for name in ("threshold", "score_thresh", "score_threshold", "conf", "confidence"):
            if name in params:
                return predict_fn(image, **{name: float(threshold)})
    except Exception:
        pass
    return predict_fn(image)


def load_class_names(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        if p.suffix.lower() == ".json":
            j = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(j, dict) and "class_names" in j:
                return list(j["class_names"])
            if isinstance(j, list):
                return [str(x) for x in j]
        with p.open("r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
            return lines
    except Exception:
        return []


def _color_for_id(i: int) -> Tuple[int, int, int]:
    r = (37 * (i + 1)) % 255
    g = (17 * (i + 1)) % 255
    b = (97 * (i + 1)) % 255
    return int(b), int(g), int(r)  # BGR for OpenCV


def overlay_masks(frame_bgr: np.ndarray, masks: List[np.ndarray], labels: List[int], alpha: float = 0.45) -> None:
    if not masks:
        return
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    for i, m in enumerate(masks):
        if m is None:
            continue
        mm = m
        if mm.shape[0] != h or mm.shape[1] != w:
            mm_u8 = (mm.astype(np.uint8) * 255)
            mm_u8 = cv2.resize(mm_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            mm = mm_u8 > 127

        color = _color_for_id(labels[i] if i < len(labels) else i)
        overlay[mm] = color
        contours, _ = cv2.findContours((mm.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_bgr, contours, -1, color, thickness=2)

    cv2.addWeighted(overlay, float(alpha), frame_bgr, float(1.0 - alpha), 0, dst=frame_bgr)


def draw_detections(frame: np.ndarray, boxes: List[List[float]], scores: List[float], labels: List[int], class_names: List[str]):
    for box, score, lab in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(max(0, round(v))) for v in box]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name = str(lab)
        if class_names and 0 <= lab < len(class_names):
            name = class_names[lab]
        text = f"{name}: {score:.2f}"
        cv2.putText(frame, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_mask_contours(frame: np.ndarray, masks: List[np.ndarray], labels: List[int]):
    # draw polygon contours for each mask for better visibility
    h, w = frame.shape[:2]
    for i, m in enumerate(masks):
        if m is None:
            continue
        mm = m.astype('uint8')
        if mm.shape[0] != h or mm.shape[1] != w:
            mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
        # find contours
        contours, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = _color_for_id(labels[i] if i < len(labels) else i)
        # draw thicker contour with black edge then color
        cv2.drawContours(frame, contours, -1, (0, 0, 0), thickness=3)
        cv2.drawContours(frame, contours, -1, color, thickness=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARIA_MoldVision: Run inference on a single image and overlay masks")
    p.add_argument("--image", "-i", required=True, help="Path to image file")
    p.add_argument("--bundle-dir", default=None, help="deployment bundle directory; preferred when available")
    p.add_argument("--weights", "-w", default=None, help="Path to model checkpoint (.pth); required when --backend pytorch")
    p.add_argument("--task", choices=["detect", "seg"], default="seg", help="Task: detect or seg (default: seg)")
    p.add_argument("--size", choices=["nano", "small", "base", "medium"], default="nano", help="Model size for detection (ignored for seg)")
    p.add_argument("--num-classes", type=int, default=None, help="Override number of classes when instantiating detect model")
    p.add_argument("--threshold", type=float, default=0.3, help="Score threshold for detections")
    p.add_argument("--mask-thresh", type=float, default=0.5, help="Mask threshold to binarize probabilities")
    p.add_argument("--mask-alpha", type=float, default=0.45, help="Alpha for mask overlay")
    p.add_argument("--classes-file", default=None, help="Path to METADATA.json or newline class file")
    p.add_argument("--device", default=None, help="Torch device (e.g. cuda:0). Default: auto")
    p.add_argument("--checkpoint-key", default=None, help="Explicit key inside checkpoint that contains state_dict")
    p.add_argument("--use-checkpoint-model", action="store_true", help="If checkpoint contains a pickled model, use it")
    p.add_argument("--dump-output", default=None, help="Path to save raw model output as JSON for inspection")
    p.add_argument("--width", type=int, default=None, help="Optional display width (pixels). Scales image for display only.")
    p.add_argument(
        "--regularize-mask",
        choices=["none", "bbox", "hull", "approx", "quad"],
        default="none",
        help="Optional mask regularization: bbox (bounding rectangle), hull (convex hull), approx (polygon approx), quad (4-point perspective quad).",
    )
    p.add_argument(
        "--regularize-eps",
        type=float,
        default=0.02,
        help="Epsilon fraction (of contour perimeter) used by approx regularization (only for --regularize-mask approx).",
    )
    p.add_argument("--output", "-o", default=None, help="Path to save overlayed output image (if omitted, will display window)")
    p.add_argument("--verbose", action="store_true")
    # ONNX backend
    p.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch",
                   help="inference backend: pytorch (default) or onnx")
    p.add_argument("--onnx-model", type=str, default=None,
                   help="path to ONNX model file (required when --backend onnx)")
    return p.parse_args()


def _looks_like_bundle_dir(path: Path) -> bool:
    required = ("model_config.json", "preprocess.json", "postprocess.json", "classes.json")
    return path.is_dir() and all((path / name).exists() for name in required)


def _infer_bundle_dir(args) -> Path | None:
    if args.bundle_dir:
        cand = Path(args.bundle_dir)
        if _looks_like_bundle_dir(cand):
            return cand
        raise FileNotFoundError(f"Bundle directory is missing required files: {cand}")

    if args.onnx_model:
        cand = Path(args.onnx_model).resolve().parent
        if _looks_like_bundle_dir(cand):
            return cand

    if args.weights:
        cand = Path(args.weights).resolve().parent
        if _looks_like_bundle_dir(cand):
            return cand

    return None


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    class_names = load_class_names(args.classes_file) if args.classes_file else []
    bundle_dir = _infer_bundle_dir(args)

    if bundle_dir is not None:
        engine = InferenceEngine(
            bundle_dir=bundle_dir,
            weights_path=(Path(args.weights) if args.weights else None),
            device=args.device,
            score_thresh=float(args.threshold),
            mask_thresh=float(args.mask_thresh),
            checkpoint_key=args.checkpoint_key,
            use_checkpoint_model=bool(args.use_checkpoint_model),
            strict=False,
            backend=args.backend,
        )
        if not class_names:
            class_names = list(engine.class_names)

        img_p = Path(args.image)
        if not img_p.exists():
            raise SystemExit(f"Image not found: {img_p}")
        frame_bgr = cv2.imread(str(img_p))
        if frame_bgr is None:
            raise SystemExit(f"Failed to read image: {img_p}")

        res = engine.infer(frame_bgr)
        boxes = list(res.boxes or [])
        scores = list(res.scores or [])
        labels = list(res.labels or [])
        masks = list(res.masks or []) if res.masks is not None else None

        disp = frame_bgr.copy()
        if args.task == "seg" and masks:
            overlay_masks(disp, masks, labels, alpha=args.mask_alpha)
        draw_detections(disp, boxes, scores, labels, class_names)

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), disp)
            print(f"Wrote overlay to: {out_p}")
        else:
            cv2.imshow("ARIA_MoldVision", disp)
            print("Press any key in the image window to exit")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        det_json = detections_to_json(boxes, scores, labels, class_names=class_names, image_id=str(img_p.name), score_thresh=args.threshold)
        print(json.dumps(det_json, indent=2))
        return

    if args.backend == "onnx":
        if not args.onnx_model:
            print("Error: --onnx-model is required when --backend onnx")
            return
        session = load_onnx_session(args.onnx_model, args.device)
        target_h, target_w = onnx_input_hw(session)

        img_p = Path(args.image)
        if not img_p.exists():
            raise SystemExit(f"Image not found: {img_p}")
        frame_bgr = cv2.imread(str(img_p))
        if frame_bgr is None:
            raise SystemExit(f"Failed to read image: {img_p}")

        boxes, scores, labels, masks = run_onnx_frame(
            session, frame_bgr, target_h, target_w,
            args.threshold, args.task == "seg", args.mask_thresh,
        )

        disp = frame_bgr.copy()
        if args.task == "seg" and masks:
            overlay_masks(disp, masks, labels, alpha=args.mask_alpha)
        draw_detections(disp, boxes, scores, labels, class_names)

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), disp)
            print(f"Wrote overlay to: {out_p}")
        else:
            cv2.imshow("ARIA_MoldVision", disp)
            print("Press any key in the image window to exit")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        det_json = detections_to_json(boxes, scores, labels, class_names=class_names, image_id=str(img_p.name), score_thresh=args.threshold)
        print(json.dumps(det_json, indent=2))
        return

    if not args.weights:
        print("Error: --weights is required when --backend pytorch")
        return

    ckpt_args = read_checkpoint_args(args.weights)
    model_size = args.size
    if args.task == "seg":
        detected_size = detect_model_size_from_checkpoint(args.weights, checkpoint_key=args.checkpoint_key)
        if detected_size:
            model_size = detected_size

    model_num_classes = args.num_classes
    if model_num_classes is None and class_names:
        model_num_classes = int(len(class_names))
    if model_num_classes is None and ckpt_args is not None:
        ck_num_classes = getattr(ckpt_args, "num_classes", None)
        if isinstance(ck_num_classes, int) and ck_num_classes > 0:
            model_num_classes = int(ck_num_classes)

    print("Instantiating model...")
    model = instantiate_model(size=model_size, num_classes=model_num_classes, task=args.task)

    ok, replacement = load_checkpoint_weights(model, args.weights, device, checkpoint_key=args.checkpoint_key, allow_replace_model=args.use_checkpoint_model, verbose=args.verbose)
    if not ok and replacement is None:
        print("Warning: could not load weights; inference may fail or use random weights")
    if replacement is not None:
        model = replacement

    # move module to device
    try:
        if hasattr(model, "to"):
            model.to(device)
    except Exception:
        pass
    try:
        if hasattr(model, "eval"):
            model.eval()
    except Exception:
        pass
    try:
        optimize = getattr(model, "optimize_for_inference", None)
        if callable(optimize):
            optimize(compile=False, batch_size=1)
    except Exception:
        pass

    img_p = Path(args.image)
    if not img_p.exists():
        raise SystemExit(f"Image not found: {img_p}")

    frame = cv2.imread(str(img_p))
    if frame is None:
        raise SystemExit(f"Failed to read image: {img_p}")
    # normalize channel count: ensure BGR (3 channels)
    if frame.ndim == 2:
        # grayscale -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        # BGRA -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = frame.shape[:2]
    if args.verbose:
        print(f"Image read: {img_p} shape={frame.shape}")

    # preprocess: ensure PIL RGB and tensor shape
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pil = pil.convert("RGB")
    tensor = T.ToTensor()(pil).unsqueeze(0)
    # ensure 3 channels in tensor; if grayscale, replicate channels
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        if args.verbose:
            print("Input tensor has 1 channel; replicating to 3 channels for RGB model input")
        tensor = tensor.repeat(1, 3, 1, 1)
    tensor = tensor.to(device)
    if args.verbose:
        print(f"frame.shape={frame.shape}, pil.mode={pil.mode}, tensor.shape={tuple(tensor.shape)}, tensor.dtype={tensor.dtype}")

    # pick torch module for device placement (reuse logic from infer_webcam)
    module_for_torch = None
    if hasattr(model, "to") and callable(getattr(model, "to")):
        module_for_torch = model
    else:
        for attr in ("model", "net", "network", "module", "detector", "backbone"):
            inner = getattr(model, attr, None)
            if inner is not None and hasattr(inner, "to") and callable(getattr(inner, "to")):
                module_for_torch = inner
                break

    if module_for_torch is not None:
        try:
            module_for_torch.to(device)
        except Exception:
            pass
        try:
            module_for_torch.eval()
        except Exception:
            pass

    # Prefer wrapper-level predict() if present (often includes correct preprocessing/postprocessing).
    out = None
    try:
        pred = getattr(model, "predict", None)
        if callable(pred):
            out = _call_predict_best_effort(pred, pil, threshold=float(args.threshold))
    except Exception:
        out = None

    # fallback: inference via shared helper to support wrapper objects
    if out is None:
        try:
            with torch.no_grad():
                out = run_inference(model, module_for_torch, tensor)
        except Exception as e:
            if args.verbose:
                print(f"Inference failed: {e}")
            out = None

    # debug: print a summary of raw model output when verbose
    if args.verbose:
        try:
            import torch as _torch

            def _summarize(o):
                if o is None:
                    return "None"
                if isinstance(o, dict):
                    return f"dict(keys={list(o.keys())})"
                if isinstance(o, (_torch.Tensor,)):
                    return f"Tensor(shape={tuple(o.shape)}, dtype={o.dtype})"
                if isinstance(o, (list, tuple)):
                    return f"{type(o).__name__}(len={len(o)})"
                try:
                    return repr(o)[:500]
                except Exception:
                    return str(type(o))

            print("Raw model output summary:", _summarize(out))
            if isinstance(out, dict):
                for k, v in out.items():
                    print(f"  key={k}: {_summarize(v)}")
            elif isinstance(out, (list, tuple)):
                for i, v in enumerate(out):
                    print(f"  out[{i}]: {_summarize(v)}")
        except Exception as _e:
            print("Failed to summarize model output:", _e)

    boxes, scores, labels, masks = parse_model_output(
        out,
        w,
        h,
        score_thresh=args.threshold,
        return_masks=True,
        mask_thresh=args.mask_thresh,
    )

    # Optionally dump raw model output and detailed tensor/key summaries for debugging
    if args.dump_output:
        import numpy as _np
        import torch as _torch

        def _tensor_summary(tensor, max_samples=10):
            try:
                t = tensor.detach().cpu()
                arr = t.numpy()
                flat = arr.ravel()
                n = flat.size
                sample = flat[:max_samples].tolist() if n > 0 else []
                summary = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "size": int(n),
                    "min": float(flat.min()) if n > 0 else None,
                    "max": float(flat.max()) if n > 0 else None,
                    "mean": float(flat.mean()) if n > 0 else None,
                    "std": float(flat.std()) if n > 0 else None,
                    "sample": sample,
                }
                return summary
            except Exception as e:
                return {"error": str(e), "type": str(type(tensor))}

        def _ndarray_summary(arr, max_samples=10):
            try:
                a = _np.asarray(arr)
                flat = a.ravel()
                n = flat.size
                sample = flat[:max_samples].tolist() if n > 0 else []
                return {
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                    "size": int(n),
                    "min": float(flat.min()) if n > 0 else None,
                    "max": float(flat.max()) if n > 0 else None,
                    "mean": float(flat.mean()) if n > 0 else None,
                    "std": float(flat.std()) if n > 0 else None,
                    "sample": sample,
                }
            except Exception as e:
                return {"error": str(e), "type": str(type(arr))}

        def _summarize_obj(obj, name="obj"):
            # return a JSON-serializable summary and a small serialized value if small
            try:
                if obj is None:
                    return {"summary": "None", "value": None}
                if isinstance(obj, (_torch.Tensor,)):
                    s = _tensor_summary(obj)
                    return {"summary": s, "value": None}
                if isinstance(obj, (_np.ndarray,)):
                    s = _ndarray_summary(obj)
                    return {"summary": s, "value": None}
                if isinstance(obj, dict):
                    keys = list(obj.keys())
                    vals = {str(k): _summarize_obj(v, name=str(k))["summary"] for k, v in obj.items()} if len(keys) <= 50 else None
                    return {"summary": {"type": "dict", "keys": keys}, "value": vals}
                if isinstance(obj, (list, tuple)):
                    ln = len(obj)
                    if ln <= 50:
                        items = [_summarize_obj(v, name=f"{name}[{i}]")["summary"] for i, v in enumerate(obj)]
                    else:
                        items = None
                    return {"summary": {"type": type(obj).__name__, "len": ln}, "value": items}
                # fallback: inspect attributes if available
                attr_keys = []
                try:
                    attr_keys = [k for k in dir(obj) if not k.startswith("_")][:200]
                except Exception:
                    attr_keys = []
                attrs = {}
                for k in ("pred_logits", "pred_boxes", "pred_masks", "masks", "mask", "scores", "scores_", "boxes"):
                    if hasattr(obj, k):
                        try:
                            v = getattr(obj, k)
                            if isinstance(v, (_torch.Tensor,)):
                                attrs[k] = _tensor_summary(v)
                            else:
                                attrs[k] = _ndarray_summary(v)
                        except Exception as e:
                            attrs[k] = {"error": str(e)}
                return {"summary": {"type": str(type(obj)), "attrs_inspected": list(attrs.keys()), "dir_sample": attr_keys[:50]}, "value": attrs}
            except Exception as e:
                return {"summary": {"error": str(e), "type": str(type(obj))}, "value": None}

        try:
            dump_p = Path(args.dump_output)
            dump_p.parent.mkdir(parents=True, exist_ok=True)

            probes = {}
            probes["out"] = _summarize_obj(out, name="out")

            # Always attempt to call the underlying torch module and the model directly
            # to capture raw logits/boxes/masks even if the wrapper returns a filtered object.
            if module_for_torch is not None:
                try:
                    if args.verbose:
                        print("Probe: calling module_for_torch(tensor) to capture raw forward outputs...")
                    with torch.no_grad():
                        direct = module_for_torch(tensor)
                    probes["module_forward"] = _summarize_obj(direct, name="module_forward")
                except Exception as e:
                    probes["module_forward_error"] = str(e)

            try:
                if callable(model):
                    if args.verbose:
                        print("Probe: calling model(tensor) to capture raw forward outputs...")
                    with torch.no_grad():
                        direct_model = model(tensor)
                    probes["model_forward"] = _summarize_obj(direct_model, name="model_forward")
                else:
                    probes["model_forward_skipped"] = "model is not callable"
            except Exception as e:
                probes["model_forward_error"] = probes.get("model_forward_error", "") + " | " + str(e)

            # write the probe summaries to disk
            # inspect likely attributes on the model and module_for_torch for raw tensors
            try:
                probes["model_attrs"] = {}
                inspect_targets = [model]
                if module_for_torch is not None and module_for_torch is not model:
                    inspect_targets.append(module_for_torch)
                attr_candidates = [
                    "last_output",
                    "last_preds",
                    "last_result",
                    "last_outputs",
                    "outputs",
                    "output",
                    "pred_logits",
                    "pred_boxes",
                    "pred_masks",
                    "logits",
                    "boxes",
                    "masks",
                    "detections",
                    "results",
                    "result",
                    "decoder_outputs",
                    "enc_out",
                    "transformer",
                    "module",
                ]
                for tgt in inspect_targets:
                    tname = type(tgt).__name__
                    probes["model_attrs"][tname] = {}
                    for an in attr_candidates:
                        if hasattr(tgt, an):
                            try:
                                v = getattr(tgt, an)
                                probes["model_attrs"][tname][an] = _summarize_obj(v, name=f"{tname}.{an}")
                            except Exception as e:
                                probes["model_attrs"][tname][an] = {"error": str(e)}
            except Exception:
                pass

            import json as _json
            dump_p.write_text(_json.dumps({"probes": probes}, indent=2), encoding="utf-8")
            if args.verbose:
                print(f"Wrote raw model probe summary to: {dump_p}")
        except Exception as e:
            import sys
            print(f"Failed to write detailed dump: {e}", file=sys.stderr)

    # If masks are missing or empty, try to extract masks directly from raw model output
    def _to_bool_masks(mobj, mask_thresh: float):
        import torch as _torch
        import numpy as _np
        if mobj is None:
            return None
        # Torch tensor
        try:
            if isinstance(mobj, _torch.Tensor):
                arr = mobj.detach().cpu().numpy()
            else:
                arr = _np.asarray(mobj)
        except Exception:
            return None

        # arr shapes: (N,H,W) or (H,W) or (N,1,H,W)
        if arr.ndim == 2:
            return [arr > mask_thresh]
        if arr.ndim == 3:
            # (N,H,W) or (1,H,W)
            if arr.shape[0] == 1:
                return [arr[0] > mask_thresh]
            return [(arr[i] > mask_thresh) for i in range(arr.shape[0])]
        if arr.ndim == 4 and arr.shape[1] == 1:
            return [(arr[i, 0] > mask_thresh) for i in range(arr.shape[0])]
        return None

    if args.task == "seg":
        need_masks = False
        if masks is None:
            need_masks = True
        else:
            # check if all masks are empty
            try:
                any_true = any(m.any() for m in masks if m is not None)
                if not any_true:
                    need_masks = True
            except Exception:
                need_masks = True

        if need_masks:
            if args.verbose:
                print("Attempting to extract masks directly from raw model output...")
            mask_candidate = None
            # dict-like
            if isinstance(out, dict):
                for k in ("masks", "mask", "pred_masks", "segmentation"):
                    if k in out:
                        mask_candidate = out[k]
                        if args.verbose:
                            print(f"Found mask candidate key in dict: {k}")
                        break
            else:
                # object attributes
                for k in ("masks", "mask", "pred_masks", "segmentation"):
                    if hasattr(out, k):
                        mask_candidate = getattr(out, k)
                        if args.verbose:
                            print(f"Found mask candidate attr on output: {k}")
                        break
                # list/tuple of objects
                if mask_candidate is None and isinstance(out, (list, tuple)) and len(out) > 0:
                    first = out[0]
                    for k in ("masks", "mask", "pred_masks", "segmentation"):
                        if hasattr(first, k):
                            mask_candidate = getattr(first, k)
                            if args.verbose:
                                print(f"Found mask candidate on out[0]: {k}")
                            break

            if mask_candidate is not None:
                masks_try = _to_bool_masks(mask_candidate, args.mask_thresh)
                if masks_try:
                    masks = masks_try
                    if args.verbose:
                        print(f"Extracted {len(masks)} masks from raw output (threshold={args.mask_thresh})")

    # Regularize masks if requested
    def _regularize_mask_list(masks_list, method: str, eps: float):
        import numpy as _np

        if not masks_list:
            return masks_list

        out_masks = []
        for m in masks_list:
            if m is None:
                out_masks.append(None)
                continue
            mm = (m.astype('uint8') * 255)
            # find contours
            cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                out_masks.append(m)
                continue

            h, w = mm.shape[:2]
            # use uint8 for OpenCV drawing functions, convert to bool before returning
            new_mask = _np.zeros((h, w), dtype=_np.uint8)

            if method == "bbox":
                # union bounding rect of all contours
                x_min = w; y_min = h; x_max = 0; y_max = 0
                for c in cnts:
                    x, y, cw, ch = cv2.boundingRect(c)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + cw - 1)
                    y_max = max(y_max, y + ch - 1)
                if x_max >= x_min and y_max >= y_min:
                    new_mask[y_min : y_max + 1, x_min : x_max + 1] = 1

            elif method == "hull":
                # convex hull of all points
                all_pts = _np.vstack(cnts).reshape(-1, 2)
                hull = cv2.convexHull(all_pts)
                cv2.fillPoly(new_mask, [hull], 1)

            elif method == "approx":
                # approximate largest contour
                # pick largest contour by area
                largest = max(cnts, key=cv2.contourArea)
                peri = cv2.arcLength(largest, True)
                epsilon = max(1.0, peri * float(eps))
                approx = cv2.approxPolyDP(largest, epsilon, True)
                cv2.fillPoly(new_mask, [approx], 1)
            
            elif method == "quad":
                # produce a 4-point quadrilateral that preserves perspective
                # apply a small closing operation to reduce small holes/gaps
                try:
                    close_k = 5
                    k = _np.ones((close_k, close_k), dtype=_np.uint8)
                    mm_closed = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, k)
                except Exception:
                    mm_closed = mm

                cnts2, _ = cv2.findContours(mm_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts2:
                    out_masks.append(m)
                    continue
                largest = max(cnts2, key=cv2.contourArea)
                peri = cv2.arcLength(largest, True)
                epsilon = max(1.0, peri * float(eps))
                approx = cv2.approxPolyDP(largest, epsilon, True)
                if approx.shape[0] == 4:
                    pts = approx.reshape(-1, 2).astype(int)
                else:
                    rect = cv2.minAreaRect(largest)
                    box = cv2.boxPoints(rect)
                    pts = _np.array(box, dtype=int)

                cv2.fillPoly(new_mask, [pts], 1)
            else:
                out_masks.append(m)
                continue

            out_masks.append(new_mask.astype(bool))

        return out_masks

    if args.regularize_mask and args.regularize_mask != "none":
        if args.verbose:
            print(f"Applying mask regularization: {args.regularize_mask} (eps={args.regularize_eps})")
        masks = _regularize_mask_list(masks or [], args.regularize_mask, args.regularize_eps)

    # overlay and draw
    disp = frame.copy()
    if args.task == "seg" and masks:
        # masks expected as boolean numpy arrays (H,W)
        overlay_masks(disp, masks, labels, alpha=args.mask_alpha)

    draw_detections(disp, boxes, scores, labels, class_names)

    # save or show
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), disp)
        print(f"Wrote overlay to: {out_p}")
    else:
        disp_to_show = disp
        if args.width:
            # scale for display only (preserve saved output resolution)
            scale = float(args.width) / float(disp.shape[1])
            new_h = max(1, int(disp.shape[0] * scale))
            disp_to_show = cv2.resize(disp, (args.width, new_h), interpolation=cv2.INTER_AREA)
            if args.verbose:
                print(f"Scaled display to width={args.width}, height={new_h}")

        cv2.imshow("ARIA_MoldVision", disp_to_show)
        print("Press any key in the image window to exit")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # print detections as JSON
    det_json = detections_to_json(boxes, scores, labels, class_names=class_names, image_id=str(img_p.name), score_thresh=args.threshold)
    print(json.dumps(det_json, indent=2))


if __name__ == "__main__":
    main()
