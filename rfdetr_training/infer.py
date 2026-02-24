from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .deploy import (
    detections_to_json,
    letterbox_pil,
    load_bundle_config,
    parse_model_output_generic,
    unletterbox_mask,
    unletterbox_xyxy,
)


@dataclass(frozen=True)
class InferResult:
    ok: bool
    payload: Optional[Dict[str, Any]] = None
    message: str = ""
    boxes: Optional[List[List[float]]] = None
    scores: Optional[List[float]] = None
    labels: Optional[List[int]] = None
    masks: Optional[List[Any]] = None


def _instantiate_model(task: str, size: str, num_classes: Optional[int]):
    task = (task or "detect").lower().strip()
    if task == "seg":
        from rfdetr import RFDETRSegPreview  # type: ignore

        return RFDETRSegPreview()

    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium  # type: ignore

    cls = {"nano": RFDETRNano, "small": RFDETRSmall, "base": RFDETRBase, "medium": RFDETRMedium}.get(size)
    if cls is None:
        raise ValueError(f"Unknown model size: {size}")
    try:
        return cls(num_classes=int(num_classes)) if num_classes is not None else cls()
    except TypeError:
        return cls()


def infer_from_bundle(
    *,
    bundle_dir: Path,
    image_path: Path,
    weights_path: Optional[Path],
    device: Optional[str],
    score_thresh: Optional[float],
    mask_thresh: Optional[float],
    checkpoint_key: Optional[str],
    use_checkpoint_model: bool,
    strict: bool,
) -> InferResult:
    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image  # type: ignore
    except Exception as e:
        return InferResult(False, None, f"Missing dependencies for inference: {e}")

    bundle_dir = bundle_dir.expanduser().resolve()
    image_path = image_path.expanduser().resolve()
    if not bundle_dir.exists():
        return InferResult(False, None, f"Bundle dir not found: {bundle_dir}")
    if not image_path.exists():
        return InferResult(False, None, f"Image not found: {image_path}")

    cfg = load_bundle_config(bundle_dir)
    model_cfg = cfg.get("model_config.json", {}) or {}
    pre_cfg = cfg.get("preprocess.json", {}) or {}
    post_cfg = cfg.get("postprocess.json", {}) or {}
    classes_raw = cfg.get("classes.json", []) or []
    if isinstance(classes_raw, dict) and "class_names" in classes_raw:
        classes_raw = classes_raw["class_names"]
    class_names: List[str] = list(classes_raw) if isinstance(classes_raw, list) else []

    task = str(model_cfg.get("task") or "detect").strip().lower()
    size = str(model_cfg.get("size") or "nano").strip().lower()
    num_classes = int(len(class_names)) if class_names else None

    device_t = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = (weights_path.expanduser().resolve() if weights_path else (bundle_dir / "checkpoint.pth"))
    if not weights.exists():
        return InferResult(False, None, f"Weights not found: {weights}")

    try:
        model = _instantiate_model(task, size, num_classes)
    except Exception as e:
        return InferResult(False, None, f"Failed to instantiate model: {e}")

    from .checkpoints import load_checkpoint_weights

    lr = load_checkpoint_weights(
        model,
        str(weights),
        device_t,
        checkpoint_key=checkpoint_key,
        allow_replace_model=bool(use_checkpoint_model),
        strict=bool(strict),
        verbose=False,
    )
    if not lr.ok and lr.replacement_model is None:
        return InferResult(False, None, f"Failed to load weights: {lr.message}")
    if lr.replacement_model is not None:
        model = lr.replacement_model

    try:
        if hasattr(model, "to"):
            model.to(device_t)
        if hasattr(model, "eval"):
            model.eval()
    except Exception:
        pass

    pil = Image.open(str(image_path)).convert("RGB")
    orig_w, orig_h = int(pil.width), int(pil.height)

    policy = str(pre_cfg.get("resize_policy") or "letterbox").strip().lower()
    target_w = int(pre_cfg.get("target_w") or 640)
    target_h = int(pre_cfg.get("target_h") or 640)

    lb = None
    if policy == "letterbox":
        pil_in, lb = letterbox_pil(pil, target_w=target_w, target_h=target_h)
    else:
        pil_in = pil.resize((target_w, target_h))

    tensor = T.ToTensor()(pil_in).unsqueeze(0).to(device_t)

    # Run inference with a few method fallbacks.
    out = None
    last_exc: Optional[Exception] = None
    for name in ("predict", "infer", "inference", "forward", "detect"):
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
        try:
            with torch.inference_mode():
                out = model(tensor)  # type: ignore[misc]
        except Exception as e:
            last_exc = e

    if out is None:
        return InferResult(False, None, f"Inference failed: {last_exc}")

    st = float(score_thresh) if score_thresh is not None else float(post_cfg.get("score_threshold_default", 0.3))
    mt = float(mask_thresh) if mask_thresh is not None else float(post_cfg.get("mask_threshold_default", 0.5))
    want_masks = task == "seg"

    boxes, scores, labels, masks = parse_model_output_generic(
        out,
        img_w=int(target_w),
        img_h=int(target_h),
        score_thresh=float(st),
        want_masks=bool(want_masks),
        mask_thresh=float(mt),
        topk=int(post_cfg.get("topk_default", 300) or 300),
    )

    if lb is not None:
        boxes = [unletterbox_xyxy(b, lb=lb, orig_w=orig_w, orig_h=orig_h) for b in boxes]
        if want_masks and masks is not None:
            mapped_masks = []
            for m in masks:
                mm = unletterbox_mask(m, lb=lb, orig_w=orig_w, orig_h=orig_h)
                mapped_masks.append(mm)
            masks = mapped_masks

    payload = detections_to_json(
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_names=class_names,
        image_id=image_path.name,
        score_thresh=float(st),
    )
    if want_masks and masks is not None:
        # Avoid embedding large arrays by default; keep this for internal use.
        payload["masks_present"] = True

    return InferResult(True, payload, "ok", boxes=boxes, scores=scores, labels=labels, masks=masks)
