from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model_factory import instantiate_rfdetr_model
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
    model, _, _ = instantiate_rfdetr_model(task, size, num_classes=num_classes, pretrain_weights=None)
    return model


def _unwrap_for_inference(model: object):
    from .torch_compat import unwrap_torch_module

    return unwrap_torch_module(model)


def _run_onnx_inference(
    *,
    bundle_dir: Path,
    image_path: Path,
    pre_cfg: Dict[str, Any],
    post_cfg: Dict[str, Any],
    class_names: List[str],
    score_thresh: Optional[float],
    mask_thresh: Optional[float],
    device: Optional[str],
    topk: int,
) -> InferResult:
    try:
        import numpy as np
        import onnxruntime as ort  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:
        return InferResult(False, None, f"ONNX runtime not available: {e}")

    onnx_path = bundle_dir / "model.onnx"
    if not onnx_path.exists():
        return InferResult(False, None, f"ONNX model not found: {onnx_path}")

    providers = ["CPUExecutionProvider"]
    try:
        available = set(ort.get_available_providers())
    except Exception:
        available = set()
    wants_cuda = (device is None) or str(device).lower().startswith("cuda")
    if wants_cuda and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as e:
        return InferResult(False, None, f"Failed to create ONNX Runtime session: {e}")

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

    arr = np.asarray(pil_in, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]

    inputs = session.get_inputs()
    if not inputs:
        return InferResult(False, None, "ONNX model has no inputs")

    try:
        output_names = [out.name for out in session.get_outputs()]
        raw_outputs = session.run(output_names, {inputs[0].name: arr})
        out = {name: value for name, value in zip(output_names, raw_outputs)}
    except Exception as e:
        return InferResult(False, None, f"ONNX inference failed: {e}")

    st = float(score_thresh) if score_thresh is not None else float(post_cfg.get("score_threshold_default", 0.3))
    mt = float(mask_thresh) if mask_thresh is not None else float(post_cfg.get("mask_threshold_default", 0.5))
    want_masks = (str(pre_cfg.get("task") or "") == "seg")

    boxes, scores, labels, masks = parse_model_output_generic(
        out,
        img_w=int(target_w),
        img_h=int(target_h),
        score_thresh=float(st),
        want_masks=bool(want_masks),
        mask_thresh=float(mt),
        topk=int(topk),
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
    payload["inference_backend"] = "onnx"
    if want_masks and masks is not None:
        payload["masks_present"] = True

    return InferResult(True, payload, "ok:onnx", boxes=boxes, scores=scores, labels=labels, masks=masks)


def _run_pytorch_inference(
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
    topk: int,
) -> InferResult:
    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image  # type: ignore
    except Exception as e:
        return InferResult(False, None, f"Missing dependencies for PyTorch inference: {e}")

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

    try:
        module = _unwrap_for_inference(model)
    except Exception as e:
        return InferResult(False, None, f"Failed to locate torch module inside model: {e}")

    from .checkpoints import load_checkpoint_weights

    lr = load_checkpoint_weights(
        module,
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
            module = _unwrap_for_inference(model)
        except Exception as e:
            return InferResult(False, None, f"Failed to locate torch module inside checkpoint model: {e}")

    model = module

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
        topk=int(topk),
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
    payload["inference_backend"] = "pytorch"
    if want_masks and masks is not None:
        payload["masks_present"] = True

    return InferResult(True, payload, "ok:pytorch", boxes=boxes, scores=scores, labels=labels, masks=masks)


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
    backend: str = "auto",
    topk: Optional[int] = None,
) -> InferResult:
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
    topk_final = int(topk) if topk is not None else int(post_cfg.get("topk_default", 300) or 300)
    backend_norm = str(backend or "auto").strip().lower()
    onnx_path = bundle_dir / "model.onnx"

    if backend_norm not in {"auto", "onnx", "pytorch"}:
        return InferResult(False, None, f"Unsupported backend: {backend}")

    if backend_norm == "onnx" and not onnx_path.exists():
        return InferResult(False, None, f"ONNX model not found: {onnx_path}")

    if backend_norm in {"auto", "onnx"} and onnx_path.exists():
        onnx_res = _run_onnx_inference(
            bundle_dir=bundle_dir,
            image_path=image_path,
            pre_cfg={**pre_cfg, "task": task},
            post_cfg=post_cfg,
            class_names=class_names,
            score_thresh=score_thresh,
            mask_thresh=mask_thresh,
            device=device,
            topk=int(topk_final),
        )
        if onnx_res.ok or backend_norm == "onnx":
            return onnx_res

    return _run_pytorch_inference(
        bundle_dir=bundle_dir,
        image_path=image_path,
        weights_path=weights_path,
        device=device,
        score_thresh=score_thresh,
        mask_thresh=mask_thresh,
        checkpoint_key=checkpoint_key,
        use_checkpoint_model=use_checkpoint_model,
        strict=strict,
        topk=int(topk_final),
    )
