from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Letterbox:
    target_w: int
    target_h: int
    new_w: int
    new_h: int
    pad_left: int
    pad_top: int
    scale: float


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_bundle_config(bundle_dir: Path) -> Dict[str, Any]:
    """Load deployment files from a directory.

    Expected (best-effort):
      - model_config.json
      - preprocess.json
      - postprocess.json
      - classes.json
    """
    bundle_dir = bundle_dir.expanduser().resolve()
    out: Dict[str, Any] = {"bundle_dir": str(bundle_dir)}
    for name in ("model_config.json", "preprocess.json", "postprocess.json", "classes.json"):
        p = bundle_dir / name
        if p.exists():
            out[name] = _read_json(p)
    return out


def letterbox_pil(pil, *, target_w: int, target_h: int, fill: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[Any, Letterbox]:
    """Resize to fit inside (target_w,target_h) and pad (keep aspect ratio)."""
    from PIL import Image  # type: ignore

    if not isinstance(pil, Image.Image):
        raise TypeError("letterbox_pil expects a PIL.Image")

    orig_w, orig_h = int(pil.width), int(pil.height)
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Invalid image size")

    scale = min(float(target_w) / float(orig_w), float(target_h) / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (int(target_w), int(target_h)), color=fill)
    pad_left = int((target_w - new_w) // 2)
    pad_top = int((target_h - new_h) // 2)
    canvas.paste(resized, (pad_left, pad_top))

    info = Letterbox(
        target_w=int(target_w),
        target_h=int(target_h),
        new_w=int(new_w),
        new_h=int(new_h),
        pad_left=int(pad_left),
        pad_top=int(pad_top),
        scale=float(scale),
    )
    return canvas, info


def unletterbox_xyxy(
    box_xyxy: List[float],
    *,
    lb: Letterbox,
    orig_w: int,
    orig_h: int,
) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = (x1 - float(lb.pad_left)) / float(lb.scale)
    y1 = (y1 - float(lb.pad_top)) / float(lb.scale)
    x2 = (x2 - float(lb.pad_left)) / float(lb.scale)
    y2 = (y2 - float(lb.pad_top)) / float(lb.scale)

    def _clamp(v: float, lo: float, hi: float) -> float:
        return lo if v < lo else hi if v > hi else v

    x1 = _clamp(x1, 0.0, float(max(0, orig_w - 1)))
    y1 = _clamp(y1, 0.0, float(max(0, orig_h - 1)))
    x2 = _clamp(x2, 0.0, float(max(0, orig_w - 1)))
    y2 = _clamp(y2, 0.0, float(max(0, orig_h - 1)))
    return [x1, y1, x2, y2]


def unletterbox_mask(mask_hw: Any, *, lb: Letterbox, orig_w: int, orig_h: int) -> Optional[Any]:
    """Map a model-input-sized mask back to original image size."""
    try:
        import numpy as np
    except Exception:
        return None

    m = mask_hw
    try:
        if hasattr(m, "detach"):
            m = m.detach().cpu().numpy()
        else:
            m = np.asarray(m)
    except Exception:
        return None

    if m.ndim != 2:
        return None

    # Ensure mask matches the model input size.
    if int(m.shape[1]) != int(lb.target_w) or int(m.shape[0]) != int(lb.target_h):
        try:
            import cv2  # type: ignore

            m = cv2.resize(m.astype("uint8"), (int(lb.target_w), int(lb.target_h)), interpolation=cv2.INTER_NEAREST)
        except Exception:
            from PIL import Image  # type: ignore

            im = Image.fromarray(m.astype("uint8") * 255)
            im = im.resize((int(lb.target_w), int(lb.target_h)), resample=Image.NEAREST)
            m = (np.asarray(im) > 127).astype("uint8")

    x0 = int(lb.pad_left)
    y0 = int(lb.pad_top)
    x1 = int(lb.pad_left + lb.new_w)
    y1 = int(lb.pad_top + lb.new_h)
    cropped = m[y0:y1, x0:x1]

    try:
        import cv2  # type: ignore

        resized = cv2.resize(cropped.astype("uint8"), (int(orig_w), int(orig_h)), interpolation=cv2.INTER_NEAREST)
        return resized.astype(bool)
    except Exception:
        from PIL import Image  # type: ignore

        im = Image.fromarray(cropped.astype("uint8") * 255)
        im = im.resize((int(orig_w), int(orig_h)), resample=Image.NEAREST)
        return (np.asarray(im) > 127)


def _tensor_to_numpy(t: Any) -> Any:
    try:
        return t.detach().cpu().numpy()
    except Exception:
        return t


def parse_model_output_detr(
    out: Any,
    *,
    model_w: int,
    model_h: int,
    score_thresh: float,
    topk: int,
    want_masks: bool,
    mask_thresh: float,
) -> Tuple[List[List[float]], List[float], List[int], Optional[List[Any]]]:
    """Parse DETR-style raw outputs (pred_logits, pred_boxes[, pred_masks])."""
    import numpy as np

    if not isinstance(out, dict):
        return [], [], [], None
    if "pred_logits" not in out or "pred_boxes" not in out:
        return [], [], [], None

    logits = out.get("pred_logits")
    boxes = out.get("pred_boxes")
    masks = None
    for k in ("pred_masks", "masks", "mask"):
        if k in out:
            masks = out.get(k)
            break

    logits = _tensor_to_numpy(logits)
    boxes = _tensor_to_numpy(boxes)
    masks = _tensor_to_numpy(masks)
    logits = np.asarray(logits) if logits is not None else None
    boxes = np.asarray(boxes) if boxes is not None else None
    masks = np.asarray(masks) if masks is not None else None

    if logits is None or boxes is None:
        return [], [], [], None

    # allow batched outputs
    if logits.ndim == 3:
        logits0 = logits[0]
    else:
        logits0 = logits
    if boxes.ndim == 3:
        boxes0 = boxes[0]
    else:
        boxes0 = boxes

    if logits0.ndim != 2 or boxes0.ndim != 2 or boxes0.shape[-1] != 4:
        return [], [], [], None

    # Some models use a "no-object" class (softmax over C+1), while others emit
    # per-class logits without a background (often sigmoid over C).
    if int(logits0.shape[-1]) == 1:
        probs = 1.0 / (1.0 + np.exp(-logits0))
        scores_t = probs[:, 0]
        labels_t = np.zeros((scores_t.shape[0],), dtype=np.int64)
    else:
        logits_shift = logits0 - np.max(logits0, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shift)
        probs = exp_logits / np.clip(np.sum(exp_logits, axis=-1, keepdims=True), 1e-12, None)
        # common DETR convention: last class is "no-object"
        if probs.shape[-1] > 1:
            probs = probs[..., :-1]
        labels_t = np.argmax(probs, axis=-1).astype(np.int64)
        scores_t = np.max(probs, axis=-1)
    keep = scores_t >= float(score_thresh)
    if bool(np.any(keep)):
        idx = np.nonzero(keep)[0]
        scores_k = scores_t[idx]
        labels_k = labels_t[idx]
        boxes_k = boxes0[idx]
    else:
        return [], [], [], None

    # Sort by score desc and keep topk.
    order = np.argsort(-scores_k)
    if int(topk) > 0 and int(order.shape[0]) > int(topk):
        order = order[: int(topk)]
    scores_k = scores_k[order]
    labels_k = labels_k[order]
    boxes_k = boxes_k[order]

    out_boxes: List[List[float]] = []
    out_scores: List[float] = []
    out_labels: List[int] = []

    # boxes are typically cxcywh normalized in 0..1
    maxv = float(np.max(boxes_k)) if int(boxes_k.size) > 0 else 0.0
    normalized = maxv <= 1.5

    for i in range(int(scores_k.shape[0])):
        sc = float(scores_k[i])
        lab = int(labels_k[i])
        cx, cy, bw, bh = [float(x) for x in boxes_k[i].tolist()]
        if normalized:
            cx *= float(model_w)
            cy *= float(model_h)
            bw *= float(model_w)
            bh *= float(model_h)
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        out_boxes.append([x1, y1, x2, y2])
        out_scores.append(sc)
        out_labels.append(lab)

    out_masks: Optional[List[Any]] = None
    if want_masks and masks is not None and int(masks.size) > 0:
        try:
            m0 = masks[0] if masks.ndim == 4 else masks
            if m0.ndim == 3:
                m0 = m0[idx][order]
                if np.issubdtype(m0.dtype, np.floating):
                    m0 = 1.0 / (1.0 + np.exp(-m0))
                out_masks = []
                for j in range(int(m0.shape[0])):
                    out_masks.append(np.asarray(m0[j]) >= float(mask_thresh))
        except Exception:
            out_masks = None

    return out_boxes, out_scores, out_labels, out_masks


def parse_model_output_generic(
    output: Any,
    *,
    img_w: int,
    img_h: int,
    score_thresh: float,
    want_masks: bool,
    mask_thresh: float,
    topk: int = 300,
) -> Tuple[List[List[float]], List[float], List[int], Optional[List[Any]]]:
    """Parse common output formats into xyxy pixel boxes.

    Returns boxes/scores/labels and optional masks (list of HxW arrays/bools).
    """
    # 1) supervision-like objects
    try:
        modname = output.__class__.__module__ if hasattr(output, "__class__") else ""
    except Exception:
        modname = ""
    if modname.startswith("supervision") or "Detections" in getattr(output, "__class__", type(output)).__name__:
        try:
            import numpy as np

            boxes = None
            if hasattr(output, "xyxy"):
                boxes = np.asarray(getattr(output, "xyxy"))
            elif hasattr(output, "xywh"):
                tmp = np.asarray(getattr(output, "xywh"))
                if tmp.size:
                    cx = tmp[:, 0]
                    cy = tmp[:, 1]
                    w_ = tmp[:, 2]
                    h_ = tmp[:, 3]
                    x1 = cx - w_ / 2
                    y1 = cy - h_ / 2
                    x2 = cx + w_ / 2
                    y2 = cy + h_ / 2
                    boxes = np.stack([x1, y1, x2, y2], axis=1)

            scores = None
            for cand in ("confidence", "confidences", "scores", "score"):
                if hasattr(output, cand):
                    scores = np.asarray(getattr(output, cand))
                    break

            labels = None
            for cand in ("class_id", "class_ids", "labels", "category_id", "class"):
                if hasattr(output, cand):
                    labels = np.asarray(getattr(output, cand))
                    break

            masks = None
            if want_masks:
                for cand in ("mask", "masks", "segmentation", "segmentations"):
                    if hasattr(output, cand):
                        masks = getattr(output, cand)
                        break

            out_boxes: List[List[float]] = []
            out_scores: List[float] = []
            out_labels: List[int] = []
            out_masks: Optional[List[Any]] = None

            if boxes is None or boxes.size == 0:
                return [], [], [], None

            if want_masks and masks is not None:
                out_masks = []
                try:
                    import numpy as np

                    if isinstance(masks, list):
                        for m in masks:
                            mm = _tensor_to_numpy(m)
                            mm = np.asarray(mm)
                            if mm.ndim == 3 and mm.shape[0] == 1:
                                mm = mm[0]
                            if mm.ndim == 2:
                                out_masks.append(mm >= float(mask_thresh))
                    else:
                        mm = np.asarray(_tensor_to_numpy(masks))
                        if mm.ndim == 3:
                            for i in range(mm.shape[0]):
                                out_masks.append(mm[i] >= float(mask_thresh))
                except Exception:
                    out_masks = None

            for i in range(int(boxes.shape[0])):
                sc = float(scores[i]) if scores is not None and i < len(scores) else 1.0
                if sc < float(score_thresh):
                    continue
                bx = boxes[i]
                bx = [float(x) for x in bx.tolist()]
                # If normalized 0..1, scale to pixels.
                if max(bx) <= 1.0:
                    bx = [bx[0] * img_w, bx[1] * img_h, bx[2] * img_w, bx[3] * img_h]
                out_boxes.append(bx)
                out_scores.append(sc)
                lab = int(labels[i]) if labels is not None and i < len(labels) else 0
                out_labels.append(lab)

            if want_masks:
                if out_masks is not None and len(out_masks) != len(out_boxes):
                    n = min(len(out_masks), len(out_boxes))
                    out_masks = out_masks[:n]
                    out_boxes = out_boxes[:n]
                    out_scores = out_scores[:n]
                    out_labels = out_labels[:n]
                return out_boxes, out_scores, out_labels, out_masks
            return out_boxes, out_scores, out_labels, None
        except Exception:
            pass

    # 2) dict with decoded boxes/scores/labels
    try:
        import numpy as np

        out0 = output[0] if isinstance(output, (list, tuple)) else output
        if isinstance(out0, dict) and "boxes" in out0:
            b = _tensor_to_numpy(out0.get("boxes"))
            s = _tensor_to_numpy(out0.get("scores"))
            l = _tensor_to_numpy(out0.get("labels") or out0.get("classes"))
            m = None
            if want_masks:
                for k in ("masks", "mask", "pred_masks", "segmentation"):
                    if k in out0:
                        m = _tensor_to_numpy(out0[k])
                        break

            if b is not None:
                b = np.asarray(b)
                if int(b.size) == 0:
                    return [], [], [], None
                if s is None:
                    s = np.ones((b.shape[0],), dtype=np.float32)
                else:
                    s = np.asarray(s).reshape(-1)
                if l is None:
                    l = np.zeros((b.shape[0],), dtype=np.int64)
                else:
                    l = np.asarray(l).reshape(-1).astype(np.int64)

                boxes: List[List[float]] = []
                scores: List[float] = []
                labels: List[int] = []
                keep_idx: List[int] = []
                for i in range(int(b.shape[0])):
                    sc = float(s[i])
                    if sc < float(score_thresh):
                        continue
                    keep_idx.append(i)
                    boxes.append([float(x) for x in np.asarray(b[i]).tolist()])
                    scores.append(sc)
                    labels.append(int(l[i]))

                masks_out: Optional[List[Any]] = None
                if want_masks and m is not None:
                    try:
                        mm = np.asarray(m)
                        if mm.ndim == 4 and mm.shape[1] == 1:
                            mm = mm[:, 0]
                        if mm.ndim == 3:
                            masks_out = [(mm[i] >= float(mask_thresh)) for i in keep_idx if i < mm.shape[0]]
                    except Exception:
                        masks_out = None

                return boxes, scores, labels, masks_out
    except Exception:
        pass

    # 3) dict with raw DETR outputs
    try:
        out0 = output[0] if isinstance(output, (list, tuple)) else output
        if isinstance(out0, dict) and "pred_logits" in out0 and "pred_boxes" in out0:
            return parse_model_output_detr(
                out0,
                model_w=int(img_w),
                model_h=int(img_h),
                score_thresh=float(score_thresh),
                topk=int(topk),
                want_masks=bool(want_masks),
                mask_thresh=float(mask_thresh),
            )
    except Exception:
        pass

    return [], [], [], None


def detections_to_json(
    *,
    boxes: List[List[float]],
    scores: List[float],
    labels: List[int],
    class_names: Optional[List[str]],
    image_id: Optional[str],
    score_thresh: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"image_id": image_id, "detections": []}
    for i in range(len(boxes)):
        sc = float(scores[i])
        if sc < float(score_thresh):
            continue
        lid = int(labels[i])
        lname = class_names[lid] if class_names and 0 <= lid < len(class_names) else str(lid)
        out["detections"].append({"bbox": [float(x) for x in boxes[i]], "score": sc, "label_id": lid, "label_name": lname})
    return out
