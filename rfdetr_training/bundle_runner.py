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
    r = (37 * (i + 1)) % 255
    g = (17 * (i + 1)) % 255
    b = (97 * (i + 1)) % 255
    return int(r), int(g), int(b)


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
        mask_img = Image.fromarray((mm.astype(np.uint8) * int(round(alpha * 255))))
        overlay.paste((*color, int(round(alpha * 255))), (0, 0), mask_img)
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
    task = (task or "detect").lower().strip()
    if task == "seg":
        from rfdetr import RFDETRSegPreview  # type: ignore

        return RFDETRSegPreview()

    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium  # type: ignore

    cls = {"nano": RFDETRNano, "small": RFDETRSmall, "base": RFDETRBase, "medium": RFDETRMedium}.get(size)
    if cls is None:
        raise ValueError(f"Unknown size: {size}")
    try:
        return cls(num_classes=int(num_classes)) if num_classes is not None else cls()
    except TypeError:
        return cls()


def load_weights(
    model: Any,
    weights_path: Path,
    device: torch.device,
    *,
    use_checkpoint_model: bool,
    checkpoint_key: Optional[str],
    strict: bool,
) -> Any:
    ckpt = None
    try:
        ckpt = torch.load(str(weights_path), map_location=device)
    except Exception:
        ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)

    if use_checkpoint_model and isinstance(ckpt, dict):
        for cand in ("ema_model", "model"):
            if cand in ckpt and hasattr(ckpt[cand], "state_dict"):
                if strict:
                    raise RuntimeError("strict=True disallows using pickled checkpoint model objects")
                return ckpt[cand]

    state = None
    if isinstance(ckpt, dict):
        if checkpoint_key and checkpoint_key in ckpt and isinstance(ckpt[checkpoint_key], dict):
            state = ckpt[checkpoint_key]
        else:
            for key in ("model_state_dict", "state_dict", "model", "net"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
        if state is None and any(k.startswith("transformer") or k.startswith("backbone") or "class_embed" in str(k) for k in ckpt.keys()):
            state = ckpt

    if not isinstance(state, dict):
        if hasattr(model, "load"):
            model.load(str(weights_path))
            return model
        raise RuntimeError("Could not find a state_dict in checkpoint")

    target = model
    if not hasattr(target, "load_state_dict"):
        for attr in ("model", "net", "network", "module", "detector", "backbone", "transformer"):
            inner = getattr(model, attr, None)
            if inner is not None and hasattr(inner, "load_state_dict"):
                target = inner
                break

    if strict:
        target.load_state_dict(state, strict=True)
    else:
        try:
            tgt_state = target.state_dict()
            filtered = {k: v for k, v in state.items() if k in tgt_state and getattr(v, "shape", None) == getattr(tgt_state[k], "shape", None)}
        except Exception:
            filtered = state
        target.load_state_dict(filtered, strict=False)
    try:
        target.to(device)
    except Exception:
        pass
    return model


def run_inference(model: Any, tensor: torch.Tensor):
    for name in ("predict", "infer", "inference", "forward", "detect"):
        fn = getattr(model, name, None)
        if callable(fn):
            return fn(tensor)
    if callable(model):
        return model(tensor)
    raise RuntimeError("Model is not callable and has no predict/infer method")


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
    # decoded dict: boxes/scores/labels + optional masks
    if isinstance(out, dict) and "boxes" in out:
        b = out.get("boxes")
        s = out.get("scores")
        l = out.get("labels") or out.get("classes")
        m = out.get("masks") or out.get("mask") or out.get("pred_masks")
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu()
            if b.numel() == 0:
                return [], [], [], None
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
                boxes.append([float(x) for x in b[i].tolist()])
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
        masks = out.get("pred_masks") or out.get("masks") or out.get("mask")
        if isinstance(logits, torch.Tensor) and logits.ndim == 3:
            logits = logits[0]
        if isinstance(boxes, torch.Tensor) and boxes.ndim == 3:
            boxes = boxes[0]
        if isinstance(masks, torch.Tensor) and masks.ndim == 4:
            masks = masks[0]
        if not isinstance(logits, torch.Tensor) or not isinstance(boxes, torch.Tensor):
            return [], [], [], None
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

        normalized = float(boxes_k.max().item()) <= 1.5
        out_boxes: List[List[float]] = []
        out_scores: List[float] = []
        out_labels: List[int] = []
        for i in range(int(scores_k.shape[0])):
            sc = float(scores_k[i].item())
            lab = int(labels_k[i].item())
            cx, cy, bw, bh = [float(x) for x in boxes_k[i].detach().cpu().tolist()]
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
    ap.add_argument("--image", "-i", required=True, help="Path to an image")
    ap.add_argument("--weights", default=None, help="Override weights path (default: bundle/checkpoint.pth)")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, cpu (default: auto)")
    ap.add_argument("--threshold", type=float, default=None, help="Override score threshold")
    ap.add_argument("--mask-thresh", type=float, default=None, help="Override mask threshold (seg only)")
    ap.add_argument("--mask-alpha", type=float, default=None, help="Mask overlay alpha (seg only)")
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

    task = (model_cfg.get("task") or "detect").strip().lower()
    size = (model_cfg.get("size") or "nano").strip().lower()
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

    model = instantiate_model(task=task, size=size, num_classes=num_classes)
    model = load_weights(
        model,
        weights_path,
        device,
        use_checkpoint_model=bool(args.use_checkpoint_model),
        checkpoint_key=args.checkpoint_key,
        strict=bool(args.strict),
    )
    try:
        model.to(device).eval()
    except Exception:
        pass

    orig = Image.open(args.image).convert("RGB")
    if policy == "letterbox":
        pil_in, lb = letterbox(orig, tw, th)
    else:
        pil_in = orig.resize((tw, th), resample=Image.BILINEAR)
        lb = {"orig_w": int(orig.width), "orig_h": int(orig.height), "pad_left": 0, "pad_top": 0, "scale": float(tw) / float(orig.width), "target_w": tw, "target_h": th, "new_w": tw, "new_h": th}

    t = T.ToTensor()(pil_in).unsqueeze(0).to(device)

    with torch.inference_mode():
        out = run_inference(model, t)

    thresh = float(args.threshold) if args.threshold is not None else float(post_cfg.get("score_threshold_default", 0.3))
    mask_thresh = float(args.mask_thresh) if args.mask_thresh is not None else float(post_cfg.get("mask_threshold_default", 0.5))
    mask_alpha = float(args.mask_alpha) if args.mask_alpha is not None else float(post_cfg.get("mask_alpha_default", 0.45))
    topk = int(args.topk) if args.topk is not None else int(post_cfg.get("topk_default", 300))

    boxes, scores, labels, masks_t = parse_detections_and_masks(
        out,
        model_w=int(tw),
        model_h=int(th),
        score_thresh=float(thresh),
        mask_thresh=float(mask_thresh),
        topk=int(topk),
    )
    mapped_boxes = [unletterbox_xyxy(b, lb) for b in boxes]

    dets = []
    for b, s, l in zip(mapped_boxes, scores, labels):
        lid = int(l)
        lname = class_names[lid] if class_names and 0 <= lid < len(class_names) else str(lid)
        dets.append({"bbox": [float(x) for x in b], "score": float(s), "label_id": lid, "label_name": lname})
    payload = {"image_id": Path(args.image).name, "detections": dets}

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.out_json}")
    else:
        print(json.dumps(payload))

    if args.out_image:
        out_img = orig
        if task == "seg" and isinstance(masks_t, torch.Tensor):
            masks_np = unletterbox_masks(masks_t, lb=lb, out_w=int(orig.width), out_h=int(orig.height), mask_thresh=float(mask_thresh))
            out_img = overlay_masks_pil(out_img, masks_np, labels, alpha=float(mask_alpha))
        out_img = draw_boxes_pil(out_img, mapped_boxes, scores, labels, class_names)
        Path(args.out_image).parent.mkdir(parents=True, exist_ok=True)
        out_img.save(args.out_image)
        print(f"Wrote overlay: {args.out_image}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

