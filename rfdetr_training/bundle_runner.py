#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from rfdetr_training.model_factory import instantiate_rfdetr_model
from rfdetr_training.checkpoints import load_checkpoint_weights
from rfdetr_training.torch_compat import infer_backbone_patch_size, unwrap_torch_module


def _first_present(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_bundle(bundle_dir: Path) -> Dict[str, Any]:
    bundle_dir = bundle_dir.expanduser().resolve()
    return {
        "bundle_dir": str(bundle_dir),
        "model_config": _read_json(bundle_dir / "model_config.json"),
        "preprocess": _read_json(bundle_dir / "preprocess.json"),
        "postprocess": _read_json(bundle_dir / "postprocess.json"),
        "classes": _read_json(bundle_dir / "classes.json"),
    }


def letterbox(pil: Image.Image, target_w: int, target_h: int, fill=(114, 114, 114)) -> Tuple[Image.Image, Dict[str, Any]]:
    w, h = int(pil.width), int(pil.height)
    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (int(target_w), int(target_h)), color=fill)
    pad_left = int((target_w - new_w) // 2)
    pad_top = int((target_h - new_h) // 2)
    canvas.paste(resized, (pad_left, pad_top))
    info = {
        "target_w": int(target_w),
        "target_h": int(target_h),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "scale": float(scale),
        "orig_w": int(w),
        "orig_h": int(h),
    }
    return canvas, info


def unletterbox_xyxy(b: List[float], info: Dict[str, Any]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in b]
    pad_left = float(info["pad_left"])
    pad_top = float(info["pad_top"])
    scale = float(info["scale"])
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    x2 = (x2 - pad_left) / scale
    y2 = (y2 - pad_top) / scale
    ow = int(info["orig_w"])
    oh = int(info["orig_h"])
    x1 = max(0.0, min(x1, float(max(0, ow - 1))))
    y1 = max(0.0, min(y1, float(max(0, oh - 1))))
    x2 = max(0.0, min(x2, float(max(0, ow - 1))))
    y2 = max(0.0, min(y2, float(max(0, oh - 1))))
    return [x1, y1, x2, y2]


def unletterbox_masks(
    masks: torch.Tensor,
    *,
    lb: Dict[str, Any],
    out_w: int,
    out_h: int,
    mask_thresh: float,
) -> List[np.ndarray]:
    """Map model-input-sized masks to original image size (bool HxW arrays)."""
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        return []

    target_w = int(lb["target_w"])
    target_h = int(lb["target_h"])
    if int(masks.shape[-1]) != target_w or int(masks.shape[-2]) != target_h:
        m = masks.unsqueeze(1).float()
        m = F.interpolate(m, size=(target_h, target_w), mode="nearest")
        masks = m[:, 0]

    x0 = int(lb["pad_left"])
    y0 = int(lb["pad_top"])
    x1 = int(lb["pad_left"] + lb["new_w"])
    y1 = int(lb["pad_top"] + lb["new_h"])
    cropped = masks[:, y0:y1, x0:x1].unsqueeze(1).float()
    resized = F.interpolate(cropped, size=(int(out_h), int(out_w)), mode="nearest")[:, 0]
    return [(resized[i] >= float(mask_thresh)).detach().cpu().numpy() for i in range(int(resized.shape[0]))]


def _color_for_id(i: int) -> Tuple[int, int, int]:
    # Bright, readable palette (cycled by class id).
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
            mm = (np.asarray(mm_img) > 127)
        color = _color_for_id(labels[i] if i < len(labels) else i)
        # IMPORTANT: apply alpha exactly once.
        # If we set both the pasted RGBA alpha and the paste mask to `alpha`,
        # the effective opacity becomes ~alpha^2 (too faint).
        a = int(round(max(0.0, min(1.0, float(alpha))) * 255))
        mask_img = Image.fromarray((mm.astype(np.uint8) * a))
        overlay.paste((*color, 255), (0, 0), mask_img)
    return Image.alpha_composite(img, overlay).convert("RGB")


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


def instantiate_model(task: str, size: str, num_classes: Optional[int]):
    model, _, _ = instantiate_rfdetr_model(task, size, num_classes=num_classes, pretrain_weights=None)
    return model


def load_weights(
    model: Any,
    weights_path: Path,
    device: torch.device,
    *,
    use_checkpoint_model: bool,
    checkpoint_key: Optional[str],
    strict: bool,
) -> Any:
    module = unwrap_torch_module(model)
    lr = load_checkpoint_weights(
        module,
        str(weights_path),
        device,
        checkpoint_key=checkpoint_key,
        allow_replace_model=bool(use_checkpoint_model),
        strict=bool(strict),
        verbose=True,
    )
    if not lr.ok and lr.replacement_model is None:
        raise RuntimeError(f"Failed to load weights: {lr.message}")
    if lr.replacement_model is not None:
        model = lr.replacement_model
    return model


def run_inference(model: Any, tensor: torch.Tensor):
    return model(tensor)

def _call_predict_best_effort(predict_fn, image: Image.Image, *, threshold: float):
    """Call a wrapper predict() with a best-effort threshold kwarg if supported."""
    try:
        import inspect as _inspect

        sig = _inspect.signature(predict_fn)
        params = sig.parameters
        # common parameter names across wrappers
        for name in ("threshold", "score_thresh", "score_threshold", "conf", "confidence"):
            if name in params:
                return predict_fn(image, **{name: float(threshold)})
    except Exception:
        pass
    return predict_fn(image)


def _filter_degenerate(
    boxes: List[List[float]],
    scores: List[float],
    labels: List[int],
    *,
    min_box_size: float,
) -> List[int]:
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


def _apply_nms(
    boxes: List[List[float]],
    scores: List[float],
    labels: List[int],
    *,
    iou_thresh: float,
    max_dets: int,
) -> List[int]:
    if not boxes:
        return []
    if iou_thresh is None:
        return list(range(len(boxes)))
    iou = float(iou_thresh)
    if iou <= 0.0 or iou >= 1.0:
        return list(range(len(boxes)))

    def _nms_fallback(boxes_t: torch.Tensor, scores_t: torch.Tensor, *, iou: float, max_keep: int) -> List[int]:
        # Pure-torch NMS fallback (portable; slower than torchvision but OK for <= topk boxes).
        x1 = boxes_t[:, 0]
        y1 = boxes_t[:, 1]
        x2 = boxes_t[:, 2]
        y2 = boxes_t[:, 3]
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = torch.argsort(scores_t, descending=True)
        keep: List[int] = []
        mk = int(max_keep) if int(max_keep) > 0 else int(order.numel())
        while order.numel() > 0 and len(keep) < mk:
            i = int(order[0].item())
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            union = areas[i] + areas[rest] - inter
            iou_t = inter / union.clamp(min=1e-6)
            order = rest[iou_t < float(iou)]
        return keep

    tv_nms = None
    try:
        from torchvision.ops import nms as tv_nms  # type: ignore
    except Exception:
        tv_nms = None

    bt = torch.tensor(boxes, dtype=torch.float32)
    st = torch.tensor(scores, dtype=torch.float32)
    lt = torch.tensor(labels, dtype=torch.int64)

    keep_all: List[int] = []
    for cls_id in torch.unique(lt).tolist():
        cls_id = int(cls_id)
        idx = torch.nonzero(lt == cls_id, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        if tv_nms is not None:
            keep_rel = tv_nms(bt[idx], st[idx], iou)
            keep_all.extend(idx[keep_rel].tolist())
        else:
            kept_rel = _nms_fallback(bt[idx], st[idx], iou=iou, max_keep=max_dets)
            if kept_rel:
                keep_all.extend(idx[torch.tensor(kept_rel, dtype=torch.int64)].tolist())

    # Keep in descending score order.
    keep_all = sorted(keep_all, key=lambda i: float(scores[i]), reverse=True)
    md = int(max_dets) if int(max_dets) > 0 else len(keep_all)
    return keep_all[:md]


def _apply_mask_nms(
    masks: torch.Tensor,
    scores: List[float],
    *,
    iou_thresh: float,
    max_dets: int,
) -> List[int]:
    """Greedy NMS on binary masks (N,H,W) using IoU."""
    if not isinstance(masks, torch.Tensor) or masks.ndim != 3:
        return list(range(len(scores)))
    iou = float(iou_thresh)
    if iou <= 0.0 or iou >= 1.0:
        return list(range(len(scores)))

    n = int(masks.shape[0])
    if n == 0:
        return []

    # sort by score desc
    order = sorted(range(min(n, len(scores))), key=lambda i: float(scores[i]), reverse=True)
    keep: List[int] = []
    areas = masks.to(dtype=torch.float32).sum(dim=(1, 2)).clamp(min=1.0)
    md = int(max_dets) if int(max_dets) > 0 else n

    while order and len(keep) < md:
        i = int(order.pop(0))
        keep.append(i)
        if not order:
            break

        mi = masks[i].to(dtype=torch.float32)
        inter = (masks[order].to(dtype=torch.float32) * mi).sum(dim=(1, 2))
        union = areas[order] + areas[i] - inter
        ious = inter / union.clamp(min=1.0)
        order = [j for j, v in zip(order, ious.tolist()) if float(v) < iou]

    return keep


def parse_detections_and_masks(
    out: Any,
    *,
    model_w: int,
    model_h: int,
    score_thresh: float,
    mask_thresh: float,
    topk: int,
) -> Tuple[List[List[float]], List[float], List[int], Optional[torch.Tensor]]:
    """Return boxes_xyxy (model space), scores, labels, and optional masks (N,H,W) in model space."""
    # supervision-like objects (rfdetr's high-level .predict often returns these)
    try:
        modname = out.__class__.__module__ if hasattr(out, "__class__") else ""
    except Exception:
        modname = ""
    if modname.startswith("supervision") or "Detections" in getattr(out, "__class__", type(out)).__name__:
        try:
            b = None
            if hasattr(out, "xyxy"):
                b = np.asarray(getattr(out, "xyxy"))
            s = None
            for cand in ("confidence", "confidences", "scores", "score"):
                if hasattr(out, cand):
                    s = np.asarray(getattr(out, cand))
                    break
            l = None
            for cand in ("class_id", "class_ids", "labels", "category_id", "class"):
                if hasattr(out, cand):
                    l = np.asarray(getattr(out, cand))
                    break

            boxes: List[List[float]] = []
            scores: List[float] = []
            labels: List[int] = []
            if b is not None and getattr(b, "size", 0):
                for i in range(int(b.shape[0])):
                    sc = float(s[i]) if s is not None and i < len(s) else 1.0
                    if sc < float(score_thresh):
                        continue
                    bb = [float(x) for x in b[i].tolist()]
                    # Some outputs may be normalized 0..1.
                    try:
                        if max(bb) <= 1.5:
                            bb = [
                                bb[0] * float(model_w),
                                bb[1] * float(model_h),
                                bb[2] * float(model_w),
                                bb[3] * float(model_h),
                            ]
                    except Exception:
                        pass
                    boxes.append(bb)
                    scores.append(sc)
                    labels.append(int(l[i]) if l is not None and i < len(l) else 0)
            return boxes, scores, labels, None
        except Exception:
            # fall through to other formats
            pass

    # decoded dict: boxes/scores/labels + optional masks
    if isinstance(out, dict) and "boxes" in out:
        b = out.get("boxes")
        s = out.get("scores")
        l = out.get("labels") or out.get("classes")
        m = _first_present(out, ("masks", "mask", "pred_masks"))
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu()
            if b.numel() == 0:
                return [], [], [], None
            # Heuristic: some wrappers return normalized boxes (0..1) even when they are already XYXY.
            # If so, scale to model-space pixels.
            try:
                is_normalized = float(b.max().item()) <= 1.5 and float(b.min().item()) >= -0.25
            except Exception:
                is_normalized = False
            if s is None:
                s = torch.ones((b.shape[0],))
            else:
                s = s.detach().cpu()
            if l is None:
                l = torch.zeros((b.shape[0],), dtype=torch.int64)
            else:
                l = l.detach().cpu().long()

            boxes: List[List[float]] = []
            scores: List[float] = []
            labels: List[int] = []
            keep_idx: List[int] = []
            for i in range(int(b.shape[0])):
                sc = float(s[i].item())
                if sc < float(score_thresh):
                    continue
                keep_idx.append(i)
                bb = [float(x) for x in b[i].tolist()]
                if is_normalized and len(bb) == 4:
                    x0, y0, x1, y1 = bb
                    # If it already looks like XYXY, scale directly; otherwise treat as CXCYWH.
                    looks_like_xyxy = (
                        0.0 <= x0 <= 1.0
                        and 0.0 <= y0 <= 1.0
                        and 0.0 <= x1 <= 1.0
                        and 0.0 <= y1 <= 1.0
                        and x1 >= x0
                        and y1 >= y0
                    )
                    if looks_like_xyxy:
                        bb = [x0 * float(model_w), y0 * float(model_h), x1 * float(model_w), y1 * float(model_h)]
                    else:
                        cx = x0 * float(model_w)
                        cy = y0 * float(model_h)
                        bw = x1 * float(model_w)
                        bh = y1 * float(model_h)
                        bb = [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]
                boxes.append(bb)
                scores.append(sc)
                labels.append(int(l[i].item()))

            masks = None
            if isinstance(m, torch.Tensor) and keep_idx:
                mm = m.detach().cpu()
                if mm.ndim == 4 and mm.shape[1] == 1:
                    mm = mm[:, 0]
                if mm.ndim == 3:
                    mm = mm[keep_idx]
                    if mm.dtype.is_floating_point:
                        mm = torch.sigmoid(mm)
                    masks = mm
            return boxes, scores, labels, masks

    # DETR raw dict: pred_logits/pred_boxes (+ optional pred_masks)
    if isinstance(out, dict) and "pred_logits" in out and "pred_boxes" in out:
        logits = out["pred_logits"]
        boxes = out["pred_boxes"]
        masks = _first_present(out, ("pred_masks", "masks", "mask"))
        if isinstance(logits, torch.Tensor) and logits.ndim == 3:
            logits = logits[0]
        if isinstance(boxes, torch.Tensor) and boxes.ndim == 3:
            boxes = boxes[0]
        if isinstance(masks, torch.Tensor) and masks.ndim == 4:
            masks = masks[0]
        if not isinstance(logits, torch.Tensor) or not isinstance(boxes, torch.Tensor):
            return [], [], [], None
        # Some models use a "no-object" class (softmax over C+1), while others emit
        # per-class logits without a background (often sigmoid over C).
        if int(logits.shape[-1]) == 1:
            probs = torch.sigmoid(logits)
            scores_t = probs[:, 0]
            labels_t = torch.zeros((scores_t.shape[0],), dtype=torch.int64, device=scores_t.device)
        else:
            probs = torch.softmax(logits, dim=-1)
            if probs.shape[-1] > 1:
                probs = probs[..., :-1]
            scores_t, labels_t = probs.max(dim=-1)
        keep = scores_t >= float(score_thresh)
        if not bool(keep.any()):
            return [], [], [], None
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        scores_k = scores_t[idx]
        labels_k = labels_t[idx]
        boxes_k = boxes[idx]
        order = torch.argsort(scores_k, descending=True)
        if int(topk) > 0 and order.numel() > int(topk):
            order = order[: int(topk)]
        scores_k = scores_k[order]
        labels_k = labels_k[order]
        boxes_k = boxes_k[order]

        # boxes are often cxcywh normalized (0..1), but some wrappers may return xyxy.
        normalized = float(boxes_k.max().item()) <= 1.5
        out_boxes: List[List[float]] = []
        out_scores: List[float] = []
        out_labels: List[int] = []
        # Decide whether pred_boxes are xyxy or cxcywh using a simple raw-shape heuristic.
        # For cxcywh, the 3rd/4th entries are widths/heights and are often < cx/cy,
        # so many boxes would violate x2>=x1/y2>=y1 if interpreted as xyxy.
        use_xyxy = False
        try:
            b0 = boxes_k.detach().cpu().float()
            if b0.ndim == 2 and int(b0.shape[-1]) == 4 and int(b0.shape[0]) > 0:
                ok = (b0[:, 2] >= b0[:, 0]) & (b0[:, 3] >= b0[:, 1])
                ratio = float(ok.float().mean().item())
                use_xyxy = ratio >= 0.70
        except Exception:
            use_xyxy = False
        for i in range(int(scores_k.shape[0])):
            sc = float(scores_k[i].item())
            lab = int(labels_k[i].item())
            b = [float(x) for x in boxes_k[i].detach().cpu().tolist()]
            if use_xyxy:
                x1, y1, x2, y2 = b
                if normalized:
                    x1 *= float(model_w)
                    x2 *= float(model_w)
                    y1 *= float(model_h)
                    y2 *= float(model_h)
                out_boxes.append([x1, y1, x2, y2])
            else:
                cx, cy, bw, bh = b
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

        out_masks = None
        if isinstance(masks, torch.Tensor):
            mm = masks
            if mm.ndim == 4 and mm.shape[1] == 1:
                mm = mm[:, 0]
            if mm.ndim == 3:
                mm = mm[idx][order]
                if mm.dtype.is_floating_point:
                    mm = torch.sigmoid(mm)
                out_masks = mm.detach().cpu()

        return out_boxes, out_scores, out_labels, out_masks

    return [], [], [], None


def main() -> int:
    ap = argparse.ArgumentParser(description="Run inference using a portable RF-DETR bundle directory")
    ap.add_argument("--bundle-dir", default=".", help="Bundle directory (contains checkpoint.pth + *config.json)")
    ap.add_argument("--task", choices=["detect", "seg"], default=None, help="Override task (detect/seg). Default: from model_config.json")
    ap.add_argument("--size", default=None, help="Override model size preset (nano/small/base/...). Default: from model_config.json")
    ap.add_argument("--image", "-i", required=True, help="Path to an image")
    ap.add_argument("--weights", default=None, help="Override weights path (default: bundle/checkpoint.pth)")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, cpu (default: auto)")
    ap.add_argument("--threshold", type=float, default=None, help="Override score threshold")
    ap.add_argument("--mask-thresh", type=float, default=None, help="Override mask threshold (seg only)")
    ap.add_argument("--mask-alpha", type=float, default=None, help="Mask overlay alpha (seg only)")
    ap.add_argument(
        "--out-masks-dir",
        default=None,
        help="Segmentation: optional directory to write per-instance mask PNGs. If set, JSON detections will include mask_path.",
    )
    boxes_group = ap.add_mutually_exclusive_group()
    boxes_group.add_argument(
        "--boxes",
        action="store_true",
        help="Draw bounding boxes on the output image. Default: enabled for detect, disabled for seg.",
    )
    boxes_group.add_argument(
        "--no-boxes",
        action="store_true",
        help="Do not draw bounding boxes on the output image.",
    )
    ap.add_argument("--nms-iou", type=float, default=None, help="IoU threshold for NMS (0..1). Default: from postprocess.json; set <=0 to disable")
    ap.add_argument(
        "--mask-nms-iou",
        type=float,
        default=None,
        help="Segmentation: IoU threshold for mask-NMS (0..1) to remove near-duplicate masks. Default: from postprocess.json; set <=0 to disable",
    )
    ap.add_argument("--max-dets", type=int, default=None, help="Max detections after NMS/filtering. Default: from postprocess.json")
    ap.add_argument("--min-box-size", type=float, default=None, help="Drop boxes with width/height < this (pixels, original model space). Default: from postprocess.json")
    ap.add_argument("--out-json", default=None, help="Write detections JSON")
    ap.add_argument("--out-image", default=None, help="Write overlay image (PNG/JPG)")
    ap.add_argument("--use-checkpoint-model", action="store_true", help="Allow using a pickled model object from the checkpoint (trusted only)")
    ap.add_argument("--checkpoint-key", default=None, help="Explicit key containing state_dict inside checkpoint")
    ap.add_argument("--strict", action="store_true", help="Strict state_dict load (recommended for deployment correctness)")
    ap.add_argument("--topk", type=int, default=None)
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    cfg = load_bundle(bundle_dir)
    model_cfg = cfg.get("model_config", {}) or {}
    pre_cfg = cfg.get("preprocess", {}) or {}
    post_cfg = cfg.get("postprocess", {}) or {}

    task = (args.task or model_cfg.get("task") or "detect").strip().lower()
    size = (args.size or model_cfg.get("size") or "nano").strip().lower()
    class_names = cfg.get("classes", []) or []
    if isinstance(class_names, dict) and "class_names" in class_names:
        class_names = class_names["class_names"]
    class_names = list(class_names) if isinstance(class_names, list) else []
    num_classes = int(len(class_names)) if class_names else None

    tw = int(pre_cfg.get("target_w") or pre_cfg.get("width") or 640)
    th = int(pre_cfg.get("target_h") or pre_cfg.get("height") or 640)
    policy = (pre_cfg.get("resize_policy") or pre_cfg.get("policy") or "letterbox").strip().lower()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    weights_path = Path(args.weights).expanduser().resolve() if args.weights else (bundle_dir / "checkpoint.pth")
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")

    # thresholds / postprocess config (also used for wrapper predict())
    thresh = float(args.threshold) if args.threshold is not None else float(post_cfg.get("score_threshold_default", 0.3))
    mask_thresh = float(args.mask_thresh) if args.mask_thresh is not None else float(post_cfg.get("mask_threshold_default", 0.5))
    mask_alpha = float(args.mask_alpha) if args.mask_alpha is not None else float(post_cfg.get("mask_alpha_default", 0.45))
    topk = int(args.topk) if args.topk is not None else int(post_cfg.get("topk_default", 300))
    nms_iou = float(args.nms_iou) if args.nms_iou is not None else float(post_cfg.get("nms_iou_threshold_default", 0.7))
    mask_nms_iou = float(args.mask_nms_iou) if args.mask_nms_iou is not None else float(post_cfg.get("mask_nms_iou_threshold_default", 0.8))
    max_dets = int(args.max_dets) if args.max_dets is not None else int(post_cfg.get("max_dets_default", 100))
    min_box_size = float(args.min_box_size) if args.min_box_size is not None else float(post_cfg.get("min_box_size_default", 1.0))

    model_obj = instantiate_model(task=task, size=size, num_classes=num_classes)
    model_obj = load_weights(
        model_obj,
        weights_path,
        device,
        use_checkpoint_model=bool(args.use_checkpoint_model),
        checkpoint_key=args.checkpoint_key,
        strict=bool(args.strict),
    )

    # Always run inference on the real torch.nn.Module, in eval mode.
    model = unwrap_torch_module(model_obj)
    try:
        model.to(device).eval()
    except Exception:
        try:
            model.eval()
        except Exception:
            pass

    # Some backbones require input H/W divisible by patch size (e.g. seg often uses 12).
    try:
        ps = infer_backbone_patch_size(model)
    except Exception:
        ps = None
    if ps:
        adj_tw = int(tw) - (int(tw) % int(ps))
        adj_th = int(th) - (int(th) % int(ps))
        if adj_tw <= 0 or adj_th <= 0:
            raise SystemExit(f"Invalid preprocess size {tw}x{th} for patch_size={ps}")
        if adj_tw != int(tw) or adj_th != int(th):
            print(f"Note: adjusting preprocess size from {tw}x{th} to {adj_tw}x{adj_th} to satisfy patch_size={ps}.")
            tw, th = int(adj_tw), int(adj_th)

    orig = Image.open(args.image).convert("RGB")

    # Prefer wrapper-level predict() for detection when available (more likely to match upstream
    # preprocessing/postprocessing). For segmentation, many rfdetr versions either do not return
    # masks from predict() by default or require additional flags; use the raw-tensor path so we
    # can reliably decode `pred_masks` when present.
    out = None
    wrapper_used = False
    try:
        pred = getattr(model_obj, "predict", None)
        if task == "detect" and callable(pred):
            with torch.inference_mode():
                out = _call_predict_best_effort(pred, orig, threshold=float(thresh))
            wrapper_used = True
    except Exception:
        out = None
        wrapper_used = False

    if out is None:
        if policy == "letterbox":
            pil_in, lb = letterbox(orig, tw, th)
        else:
            pil_in = orig.resize((tw, th), resample=Image.BILINEAR)
            lb = {
                "orig_w": int(orig.width),
                "orig_h": int(orig.height),
                "pad_left": 0,
                "pad_top": 0,
                "scale": float(tw) / float(orig.width),
                "target_w": tw,
                "target_h": th,
                "new_w": tw,
                "new_h": th,
            }

        t = T.ToTensor()(pil_in).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = run_inference(model, t)
    else:
        # identity mapping for wrapper outputs (already in original image coordinates)
        lb = {
            "orig_w": int(orig.width),
            "orig_h": int(orig.height),
            "pad_left": 0,
            "pad_top": 0,
            "scale": 1.0,
            "target_w": int(orig.width),
            "target_h": int(orig.height),
            "new_w": int(orig.width),
            "new_h": int(orig.height),
        }

    boxes, scores, labels, masks_t = parse_detections_and_masks(
        out,
        model_w=(int(orig.width) if wrapper_used else int(tw)),
        model_h=(int(orig.height) if wrapper_used else int(th)),
        score_thresh=float(thresh),
        mask_thresh=float(mask_thresh),
        topk=int(topk),
    )

    # Drop degenerate boxes and optionally NMS before unletterboxing.
    keep = _filter_degenerate(boxes, scores, labels, min_box_size=float(min_box_size))
    if keep:
        boxes = [boxes[i] for i in keep]
        scores = [scores[i] for i in keep]
        labels = [labels[i] for i in keep]
        if (
            isinstance(masks_t, torch.Tensor)
            and masks_t.ndim == 3
            and len(keep) > 0
            and int(masks_t.shape[0]) > int(max(keep))
        ):
            masks_t = masks_t[keep]
    if boxes:
        keep2 = _apply_nms(boxes, scores, labels, iou_thresh=float(nms_iou), max_dets=int(max_dets))
        if keep2:
            boxes = [boxes[i] for i in keep2]
            scores = [scores[i] for i in keep2]
            labels = [labels[i] for i in keep2]
            if (
                isinstance(masks_t, torch.Tensor)
                and masks_t.ndim == 3
                and len(keep2) > 0
                and int(masks_t.shape[0]) > int(max(keep2))
            ):
                masks_t = masks_t[keep2]

    # Optional: further suppress near-duplicate masks for seg (raw-tensor path only).
    if (not wrapper_used) and task == "seg" and isinstance(masks_t, torch.Tensor) and masks_t.ndim == 3 and boxes:
        try:
            masks_bin = (masks_t >= float(mask_thresh))
            keepm = _apply_mask_nms(masks_bin, scores, iou_thresh=float(mask_nms_iou), max_dets=int(max_dets))
            if keepm:
                boxes = [boxes[i] for i in keepm]
                scores = [scores[i] for i in keepm]
                labels = [labels[i] for i in keepm]
                if int(masks_t.shape[0]) > int(max(keepm)):
                    masks_t = masks_t[keepm]
        except Exception:
            pass
    mapped_boxes = boxes if wrapper_used else [unletterbox_xyxy(b, lb) for b in boxes]

    dets: List[Dict[str, Any]] = []
    for b, s, l in zip(mapped_boxes, scores, labels):
        lid = int(l)
        lname = class_names[lid] if class_names and 0 <= lid < len(class_names) else str(lid)
        dets.append({"bbox": [float(x) for x in b], "score": float(s), "label_id": lid, "label_name": lname})

    payload: Dict[str, Any] = {
        "image_id": Path(args.image).name,
        "task": str(task),
        "detections": dets,
    }

    # For seg bundles, we do not inline full masks into JSON by default (too large),
    # but we do mark their presence and optionally write mask PNGs to disk.
    if task == "seg" and isinstance(masks_t, torch.Tensor):
        payload["masks_present"] = True
        payload["mask_threshold"] = float(mask_thresh)
        payload["mask_alpha"] = float(mask_alpha)

    out_json_path = str(args.out_json) if args.out_json else None

    if args.out_image:
        try:
            out_image_path = Path(args.out_image)
            if out_json_path and str(out_image_path) == str(out_json_path):
                raise SystemExit("--out-image and --out-json point to the same path; pass a real image path (e.g. out.jpg).")
            ext = out_image_path.suffix.lower()
            if ext in (".json",):
                raise SystemExit(f"--out-image must be an image path (e.g. out.jpg/out.png); got: {out_image_path}")
            if ext and ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
                raise SystemExit(
                    f"--out-image has an unknown/unsupported extension {ext!r}. Use .png or .jpg (recommended)."
                )
        except SystemExit:
            raise
        except Exception:
            # best-effort; let PIL error if needed
            pass

        out_img = orig
        if task == "seg" and isinstance(masks_t, torch.Tensor):
            masks_np = unletterbox_masks(masks_t, lb=lb, out_w=int(orig.width), out_h=int(orig.height), mask_thresh=float(mask_thresh))
            out_img = overlay_masks_pil(out_img, masks_np, labels, alpha=float(mask_alpha))
            if args.out_masks_dir:
                out_masks_dir = Path(args.out_masks_dir).expanduser().resolve()
                out_masks_dir.mkdir(parents=True, exist_ok=True)
                # Write masks as 8-bit PNG (0/255). One file per kept detection.
                for i, m in enumerate(masks_np):
                    mp = (np.asarray(m).astype(np.uint8) * 255)
                    Image.fromarray(mp).save(out_masks_dir / f"mask_{i:04d}.png")
                    if i < len(dets):
                        dets[i]["mask_path"] = str((out_masks_dir / f"mask_{i:04d}.png").as_posix())
                payload["mask_format"] = "png"

        # Default visualization: seg = masks only, detect = boxes.
        if args.no_boxes:
            draw_boxes = False
        elif args.boxes:
            draw_boxes = True
        else:
            draw_boxes = (task != "seg")

        if draw_boxes:
            out_img = draw_boxes_pil(out_img, mapped_boxes, scores, labels, class_names)
        Path(args.out_image).parent.mkdir(parents=True, exist_ok=True)
        out_img.save(args.out_image)
        print(f"Wrote overlay: {args.out_image}")

    if out_json_path:
        Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {out_json_path}")
    else:
        print(json.dumps(payload))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
