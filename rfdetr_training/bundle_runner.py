#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rfdetr_training.deploy import detections_to_json, filter_known_class_detections, load_bundle_config
from rfdetr_training.infer import infer_from_bundle


def _color_for_id(i: int) -> Tuple[int, int, int]:
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 255, 255),
    ]
    return palette[int(i) % len(palette)]


def overlay_masks_pil(base: Image.Image, masks: List[np.ndarray], labels: List[int], alpha: float) -> Image.Image:
    if not masks:
        return base
    img = base.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    w, h = img.size
    for i, m in enumerate(masks):
        if m is None:
            continue
        mm = np.asarray(m)
        if mm.ndim != 2:
            continue
        if mm.shape[0] != h or mm.shape[1] != w:
            mm_img = Image.fromarray((mm.astype(np.uint8) * 255))
            mm_img = mm_img.resize((w, h), resample=Image.NEAREST)
            mm = np.asarray(mm_img) > 127
        color = _color_for_id(labels[i] if i < len(labels) else i)
        a = int(round(max(0.0, min(1.0, float(alpha))) * 255))
        mask_img = Image.fromarray(mm.astype(np.uint8) * a)
        overlay.paste((*color, 255), (0, 0), mask_img)
    return Image.alpha_composite(img, overlay).convert("RGB")


def overlay_masks_visible(frame_bgr: np.ndarray, masks: List[Any], labels: List[int], alpha: float = 0.45) -> None:
    if not masks:
        return
    try:
        import cv2
    except ImportError:
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
            mm_u8 = mm.astype(np.uint8) * 255
            mm_u8 = cv2.resize(mm_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            mm = mm_u8 > 127
        color = _color_for_id(labels[i] if i < len(labels) else i)
        overlay[mm] = color
        contours, _ = cv2.findContours((mm.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_bgr, contours, -1, color, 2)
    cv2.addWeighted(overlay, float(alpha), frame_bgr, float(1.0 - alpha), 0, dst=frame_bgr)


def draw_boxes_pil(base: Image.Image, boxes: List[List[float]], scores: List[float], labels: List[int], class_names: List[str]) -> Image.Image:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [float(x) for x in b]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        lid = int(l)
        lname = class_names[lid] if class_names and 0 <= lid < len(class_names) else str(lid)
        txt = f"{lname}:{float(s):.2f}"
        if font is not None:
            draw.text((x1 + 2, max(0, y1 - 12)), txt, fill=(0, 255, 0), font=font)
        else:
            draw.text((x1 + 2, max(0, y1 - 12)), txt, fill=(0, 255, 0))
    return img


def _filter_degenerate(boxes: List[List[float]], scores: List[float], labels: List[int], *, min_box_size: float) -> List[int]:
    keep: List[int] = []
    mbs = float(min_box_size)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in b]
        if (x2 - x1) < mbs or (y2 - y1) < mbs:
            continue
        if not np.isfinite([x1, y1, x2, y2, float(scores[i]), float(labels[i])]).all():
            continue
        keep.append(int(i))
    return keep


def _nms_indices_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float, max_keep: int) -> List[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    order = np.argsort(-scores)
    keep: List[int] = []
    limit = int(max_keep) if int(max_keep) > 0 else int(order.shape[0])
    while order.size > 0 and len(keep) < limit:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0.0, None) * np.clip(yy2 - yy1, 0.0, None)
        union = np.clip(areas[i] + areas[rest] - inter, 1e-6, None)
        iou = inter / union
        order = rest[iou < float(iou_thresh)]
    return keep


def _apply_nms(boxes: List[List[float]], scores: List[float], labels: List[int], *, iou_thresh: float, max_dets: int) -> List[int]:
    if not boxes:
        return []
    iou = float(iou_thresh)
    if iou <= 0.0 or iou >= 1.0:
        return list(range(len(boxes)))
    boxes_np = np.asarray(boxes, dtype=np.float32)
    scores_np = np.asarray(scores, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64)
    keep_all: List[int] = []
    for cls_id in np.unique(labels_np):
        idx = np.nonzero(labels_np == cls_id)[0]
        if idx.size == 0:
            continue
        keep_rel = _nms_indices_numpy(boxes_np[idx], scores_np[idx], float(iou), int(max_dets))
        keep_all.extend(idx[np.asarray(keep_rel, dtype=np.int64)].tolist())
    keep_all = sorted(keep_all, key=lambda i: float(scores[i]), reverse=True)
    md = int(max_dets) if int(max_dets) > 0 else len(keep_all)
    return keep_all[:md]


def _apply_mask_nms(masks: List[Any], scores: List[float], *, iou_thresh: float, max_dets: int) -> List[int]:
    if not masks:
        return list(range(len(scores)))
    iou = float(iou_thresh)
    if iou <= 0.0 or iou >= 1.0:
        return list(range(len(scores)))

    mm = [np.asarray(m).astype(bool) for m in masks if m is not None]
    if not mm or len(mm) != len(scores):
        return list(range(min(len(masks), len(scores))))

    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    keep: List[int] = []
    md = int(max_dets) if int(max_dets) > 0 else len(order)

    while order and len(keep) < md:
        i = int(order.pop(0))
        keep.append(i)
        if not order:
            break
        mi = mm[i]
        next_order: List[int] = []
        for j in order:
            mj = mm[j]
            inter = float(np.logical_and(mi, mj).sum())
            union = float(np.logical_or(mi, mj).sum())
            iou_v = inter / max(union, 1.0)
            if iou_v < iou:
                next_order.append(j)
        order = next_order

    return keep


def _save_masks(masks: List[Any], labels: List[int], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for i, m in enumerate(masks):
        mm = np.asarray(m).astype(bool)
        mask_img = Image.fromarray(mm.astype(np.uint8) * 255)
        name = f"mask_{i:04d}_cls{int(labels[i]) if i < len(labels) else 0}.png"
        mask_path = out_dir / name
        mask_img.save(mask_path)
        paths.append(str(mask_path))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Run inference using a portable RF-DETR bundle directory")
    ap.add_argument("--bundle-dir", default=".", help="Bundle directory (contains model.onnx/checkpoint.pth + *config.json)")
    
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", "-i", help="Path to an image")
    input_group.add_argument("--video", "-v", help="Path to a video file")
    input_group.add_argument("--webcam", "-w", type=int, help="Webcam index (e.g. 0)")
    
    ap.add_argument("--weights", default=None, help="Override PyTorch checkpoint path (default: bundle/checkpoint.pth)")
    ap.add_argument("--backend", choices=["auto", "tensorrt", "onnx", "pytorch"], default="auto", help="Inference backend. Default: auto (prefer TensorRT, then ONNX)")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, cpu (default: auto)")
    ap.add_argument("--threshold", type=float, default=None, help="Override score threshold")
    ap.add_argument("--mask-thresh", type=float, default=None, help="Override mask threshold (seg only)")
    ap.add_argument("--mask-alpha", type=float, default=None, help="Mask overlay alpha (seg only)")
    ap.add_argument("--out-masks-dir", default=None, help="Segmentation: optional directory to write per-instance mask PNGs (image mode only)")
    
    boxes_group = ap.add_mutually_exclusive_group()
    boxes_group.add_argument("--boxes", action="store_true", help="Draw bounding boxes on the output image")
    boxes_group.add_argument("--no-boxes", action="store_true", help="Do not draw bounding boxes on the output image")
    
    ap.add_argument("--nms-iou", type=float, default=None, help="IoU threshold for NMS (0..1). Default: from postprocess.json")
    ap.add_argument("--mask-nms-iou", type=float, default=None, help="Segmentation: IoU threshold for mask NMS (0..1)")
    ap.add_argument("--max-dets", type=int, default=None, help="Max detections after NMS/filtering. Default: from postprocess.json")
    ap.add_argument("--min-box-size", type=float, default=None, help="Drop boxes with width/height < this")
    
    ap.add_argument("--out-json", default=None, help="Write detections JSON (image mode only)")
    ap.add_argument("--out-image", default=None, help="Write overlay image (image mode only)")
    ap.add_argument("--out-video", default=None, help="Write overlay video (video mode only)")
    ap.add_argument("--display", action="store_true", help="Display results in a window (video/webcam mode)")
    
    ap.add_argument("--use-checkpoint-model", action="store_true", help="Allow using a pickled model object from the checkpoint (trusted only)")
    ap.add_argument("--checkpoint-key", default=None, help="Explicit key containing state_dict inside checkpoint")
    ap.add_argument("--strict", action="store_true", help="Strict state_dict load for PyTorch fallback")
    ap.add_argument("--topk", type=int, default=None, help="Override top-k before postprocess filtering")
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    from rfdetr_training.infer import InferenceEngine
    
    engine = InferenceEngine(
        bundle_dir=bundle_dir,
        weights_path=(Path(args.weights) if args.weights else None),
        device=args.device,
        score_thresh=args.threshold,
        mask_thresh=args.mask_thresh,
        checkpoint_key=args.checkpoint_key,
        use_checkpoint_model=bool(args.use_checkpoint_model),
        strict=bool(args.strict),
        backend=args.backend,
        topk=args.topk,
    )

    model_cfg = engine.model_cfg
    post_cfg = engine.post_cfg
    class_names = engine.class_names
    task = engine.task

    mask_alpha = float(args.mask_alpha) if args.mask_alpha is not None else float(post_cfg.get("mask_alpha_default", 0.45))
    nms_iou = float(args.nms_iou) if args.nms_iou is not None else float(post_cfg.get("nms_iou_threshold_default", 0.7))
    mask_nms_iou = float(args.mask_nms_iou) if args.mask_nms_iou is not None else float(post_cfg.get("mask_nms_iou_threshold_default", 0.8))
    max_dets = int(args.max_dets) if args.max_dets is not None else int(post_cfg.get("max_dets_default", 100))
    min_box_size = float(args.min_box_size) if args.min_box_size is not None else float(post_cfg.get("min_box_size_default", 1.0))

    def process_frame(frame_np):
        res = engine.infer(frame_np)
        if not res.ok or res.boxes is None:
            return None, None
        
        boxes = list(res.boxes)
        scores = list(res.scores)
        labels = list(res.labels)
        masks = list(res.masks) if res.masks is not None else None

        keep = _filter_degenerate(boxes, scores, labels, min_box_size=float(min_box_size))
        if keep:
            boxes = [boxes[i] for i in keep]
            scores = [scores[i] for i in keep]
            labels = [labels[i] for i in keep]
            if masks is not None:
                masks = [masks[i] for i in keep if i < len(masks)]
        else:
            boxes, scores, labels = [], [], []
            masks = [] if masks is not None else None

        keep = _apply_nms(boxes, scores, labels, iou_thresh=float(nms_iou), max_dets=int(max_dets))
        boxes = [boxes[i] for i in keep]
        scores = [scores[i] for i in keep]
        labels = [labels[i] for i in keep]
        if masks is not None:
            masks = [masks[i] for i in keep if i < len(masks)]
            keep_masks = _apply_mask_nms(masks, scores, iou_thresh=float(mask_nms_iou), max_dets=int(max_dets))
            boxes = [boxes[i] for i in keep_masks]
            scores = [scores[i] for i in keep_masks]
            labels = [labels[i] for i in keep_masks]
            masks = [masks[i] for i in keep_masks]

        boxes, scores, labels, masks = filter_known_class_detections(
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=class_names,
            masks=masks,
        )
            
        return res, (boxes, scores, labels, masks)

    if args.image:
        img_path = Path(args.image)
        orig = Image.open(img_path).convert("RGB")
        frame_np = np.asarray(orig)[:, :, ::-1] # BGR for consistency
        res, (boxes, scores, labels, masks) = process_frame(frame_np)
        
        if res is None:
            raise SystemExit("Inference failed")

        payload = detections_to_json(
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=class_names,
            image_id=img_path.name,
            score_thresh=engine.score_thresh,
        )
        payload["inference_backend"] = res.payload.get("inference_backend")

        if masks is not None:
            payload["masks_present"] = True
            if args.out_masks_dir:
                mask_paths = _save_masks(masks, labels, Path(args.out_masks_dir))
                for det, mask_path in zip(payload["detections"], mask_paths):
                    det["mask_path"] = mask_path

        if args.out_json:
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Wrote JSON: {args.out_json}")
        else:
            print(json.dumps(payload))

        if args.out_image:
            draw_boxes = bool(args.boxes) or (not bool(args.no_boxes) and task != "seg")
            out_img = orig
            if masks:
                out_img = overlay_masks_pil(out_img, [np.asarray(m) for m in masks], labels, alpha=float(mask_alpha))
            if draw_boxes:
                out_img = draw_boxes_pil(out_img, boxes, scores, labels, class_names)
            Path(args.out_image).parent.mkdir(parents=True, exist_ok=True)
            out_img.save(args.out_image)
            print(f"Wrote overlay: {args.out_image}")

    elif args.video or args.webcam is not None:
        try:
            import cv2
        except ImportError:
            raise SystemExit("OpenCV is required for video/webcam inference. Install it with: pip install opencv-python")

        cap = cv2.VideoCapture(args.video if args.video else args.webcam)
        if not cap.isOpened():
            raise SystemExit(f"Failed to open {'video' if args.video else 'webcam'}")

        writer = None
        if args.out_video:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                res, (boxes, scores, labels, masks) = process_frame(frame)
                if res is None:
                    continue
                
                # Draw results on frame for display/saving
                disp = frame.copy()
                if masks:
                    overlay_masks_visible(disp, masks, labels, alpha=float(mask_alpha))
                
                draw_boxes = bool(args.boxes) or (not bool(args.no_boxes) and task != "seg")
                if draw_boxes:
                    for b, s, l in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = [int(v) for v in b]
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        lname = class_names[l] if class_names and 0 <= l < len(class_names) else str(l)
                        cv2.putText(disp, f"{lname}:{s:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if writer:
                    writer.write(disp)
                
                if args.display:
                    cv2.imshow("RF-DETR Inference", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            if args.display:
                cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
