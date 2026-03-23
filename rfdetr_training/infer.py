from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .model_factory import instantiate_rfdetr_model
from .deploy import (
    detections_to_json,
    filter_known_class_detections,
    letterbox_pil,
    load_bundle_config,
    normalize_image_nchw,
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


def _adjust_dims_to_patch_size(*, target_h: int, target_w: int, patch_size: Optional[int]) -> tuple[int, int]:
    if not patch_size:
        return int(target_h), int(target_w)
    h = int(target_h)
    w = int(target_w)
    adj_h = h - (h % int(patch_size))
    adj_w = w - (w % int(patch_size))
    if adj_h <= 0 or adj_w <= 0:
        return h, w
    return adj_h, adj_w


def _run_tensorrt_inference(
    *,
    bundle_dir: Path,
    image_path: Path,
    pre_cfg: Dict[str, Any],
    post_cfg: Dict[str, Any],
    class_names: List[str],
    score_thresh: Optional[float],
    mask_thresh: Optional[float],
    topk: int,
) -> InferResult:
    try:
        import numpy as np
        import tensorrt as trt  # type: ignore
        import pycuda.autoinit  # type: ignore  # noqa: F401
        import pycuda.driver as cuda  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:
        return InferResult(False, None, f"TensorRT runtime not available: {e}")

    engine_path = bundle_dir / "model.engine"
    if not engine_path.exists():
        return InferResult(False, None, f"TensorRT engine not found: {engine_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    try:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    except Exception as e:
        return InferResult(False, None, f"Failed to load TensorRT engine: {e}")
    if engine is None:
        return InferResult(False, None, f"Failed to deserialize TensorRT engine: {engine_path}")

    try:
        context = engine.create_execution_context()
    except Exception as e:
        return InferResult(False, None, f"Failed to create TensorRT execution context: {e}")
    if context is None:
        return InferResult(False, None, "TensorRT execution context is null")

    use_tensor_api = hasattr(engine, "num_io_tensors") and hasattr(context, "set_tensor_address")
    tensor_meta: List[Dict[str, Any]] = []
    try:
        if use_tensor_api:
            count = int(engine.num_io_tensors)
            for idx in range(count):
                name = str(engine.get_tensor_name(idx))
                mode = engine.get_tensor_mode(name)
                tensor_meta.append(
                    {
                        "index": idx,
                        "name": name,
                        "is_input": bool(mode == trt.TensorIOMode.INPUT),
                        "dtype": np.dtype(trt.nptype(engine.get_tensor_dtype(name))),
                        "shape": tuple(int(v) for v in engine.get_tensor_shape(name)),
                    }
                )
        else:
            count = int(engine.num_bindings)
            for idx in range(count):
                tensor_meta.append(
                    {
                        "index": idx,
                        "name": str(engine.get_binding_name(idx)),
                        "is_input": bool(engine.binding_is_input(idx)),
                        "dtype": np.dtype(trt.nptype(engine.get_binding_dtype(idx))),
                        "shape": tuple(int(v) for v in engine.get_binding_shape(idx)),
                    }
                )
    except Exception as e:
        return InferResult(False, None, f"Failed to inspect TensorRT engine bindings: {e}")

    inputs_meta = [m for m in tensor_meta if m["is_input"]]
    outputs_meta = [m for m in tensor_meta if not m["is_input"]]
    if not inputs_meta:
        return InferResult(False, None, "TensorRT engine has no inputs")
    if not outputs_meta:
        return InferResult(False, None, "TensorRT engine has no outputs")

    input_meta = inputs_meta[0]
    policy = str(pre_cfg.get("resize_policy") or "letterbox").strip().lower()
    target_w = int(pre_cfg.get("target_w") or 640)
    target_h = int(pre_cfg.get("target_h") or 640)
    input_shape = tuple(int(v) for v in input_meta["shape"])
    if len(input_shape) >= 4:
        if input_shape[2] > 0:
            target_h = int(input_shape[2])
        if input_shape[3] > 0:
            target_w = int(input_shape[3])

    try:
        desired_shape = (1, 3, int(target_h), int(target_w))
        if any(v <= 0 for v in input_shape):
            if use_tensor_api and hasattr(context, "set_input_shape"):
                context.set_input_shape(str(input_meta["name"]), desired_shape)
            elif hasattr(context, "set_binding_shape"):
                context.set_binding_shape(int(input_meta["index"]), desired_shape)
    except Exception as e:
        return InferResult(False, None, f"Failed to configure TensorRT input shape: {e}")

    pil = Image.open(str(image_path)).convert("RGB")
    orig_w, orig_h = int(pil.width), int(pil.height)
    lb = None
    if policy == "letterbox":
        pil_in, lb = letterbox_pil(pil, target_w=target_w, target_h=target_h)
    else:
        pil_in = pil.resize((target_w, target_h))

    arr = np.asarray(pil_in, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    arr = normalize_image_nchw(arr)
    arr = np.ascontiguousarray(arr.astype(input_meta["dtype"], copy=False))

    host_buffers: Dict[str, Any] = {}
    device_buffers: Dict[str, Any] = {}
    stream = cuda.Stream()
    bindings: List[int] = [0] * len(tensor_meta)

    try:
        for meta in tensor_meta:
            name = str(meta["name"])
            idx = int(meta["index"])
            if meta["is_input"]:
                host = arr
            else:
                if use_tensor_api and hasattr(context, "get_tensor_shape"):
                    shape = tuple(int(v) for v in context.get_tensor_shape(name))
                else:
                    shape = tuple(int(v) for v in context.get_binding_shape(idx))
                if any(v < 0 for v in shape):
                    return InferResult(False, None, f"TensorRT output shape unresolved for {name}: {shape}")
                host = np.empty(shape, dtype=meta["dtype"])
            host = np.ascontiguousarray(host)
            device_mem = cuda.mem_alloc(int(host.nbytes))
            host_buffers[name] = host
            device_buffers[name] = device_mem
            bindings[idx] = int(device_mem)
            if meta["is_input"]:
                cuda.memcpy_htod_async(device_mem, host, stream)
            if use_tensor_api:
                context.set_tensor_address(name, int(device_mem))
    except Exception as e:
        return InferResult(False, None, f"Failed to allocate TensorRT buffers: {e}")

    try:
        if use_tensor_api and hasattr(context, "execute_async_v3"):
            ok = bool(context.execute_async_v3(stream.handle))
        elif hasattr(context, "execute_async_v2"):
            ok = bool(context.execute_async_v2(bindings=bindings, stream_handle=stream.handle))
        elif hasattr(context, "execute_v2"):
            ok = bool(context.execute_v2(bindings=bindings))
        else:
            return InferResult(False, None, "TensorRT execution API not supported by this runtime")
    except Exception as e:
        return InferResult(False, None, f"TensorRT inference failed: {e}")
    if not ok:
        return InferResult(False, None, "TensorRT inference returned failure status")

    try:
        for meta in outputs_meta:
            name = str(meta["name"])
            cuda.memcpy_dtoh_async(host_buffers[name], device_buffers[name], stream)
        stream.synchronize()
    except Exception as e:
        return InferResult(False, None, f"Failed to read TensorRT outputs: {e}")

    out = {str(meta["name"]): host_buffers[str(meta["name"])] for meta in outputs_meta}
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
            masks = [unletterbox_mask(m, lb=lb, orig_w=orig_w, orig_h=orig_h, mask_thresh=float(mt)) for m in masks]

    payload = detections_to_json(
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_names=class_names,
        image_id=image_path.name,
        score_thresh=float(st),
    )
    payload["inference_backend"] = "tensorrt"
    if want_masks and masks is not None:
        payload["masks_present"] = True

    return InferResult(True, payload, "ok:tensorrt", boxes=boxes, scores=scores, labels=labels, masks=masks)


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

    inputs = session.get_inputs()
    if not inputs:
        return InferResult(False, None, "ONNX model has no inputs")
    try:
        input_shape = inputs[0].shape
        if len(input_shape) >= 4:
            h_dim = input_shape[2]
            w_dim = input_shape[3]
            if isinstance(h_dim, int) and h_dim > 0:
                target_h = int(h_dim)
            if isinstance(w_dim, int) and w_dim > 0:
                target_w = int(w_dim)
    except Exception:
        pass

    lb = None
    if policy == "letterbox":
        pil_in, lb = letterbox_pil(pil, target_w=target_w, target_h=target_h)
    else:
        pil_in = pil.resize((target_w, target_h))

    arr = np.asarray(pil_in, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    arr = normalize_image_nchw(arr).astype(np.float32, copy=False)

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
                mm = unletterbox_mask(m, lb=lb, orig_w=orig_w, orig_h=orig_h, mask_thresh=float(mt))
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

    from .torch_compat import infer_backbone_patch_size

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
    try:
        optimize = getattr(model, "optimize_for_inference", None)
        if callable(optimize):
            optimize(compile=False, batch_size=1)
    except Exception:
        pass

    pil = Image.open(str(image_path)).convert("RGB")
    orig_w, orig_h = int(pil.width), int(pil.height)

    policy = str(pre_cfg.get("resize_policy") or "letterbox").strip().lower()
    target_w = int(pre_cfg.get("target_w") or 640)
    target_h = int(pre_cfg.get("target_h") or 640)
    patch_size = infer_backbone_patch_size(module)
    target_h, target_w = _adjust_dims_to_patch_size(target_h=target_h, target_w=target_w, patch_size=patch_size)

    lb = None
    if policy == "letterbox":
        pil_in, lb = letterbox_pil(pil, target_w=target_w, target_h=target_h)
    else:
        pil_in = pil.resize((target_w, target_h))

    st = float(score_thresh) if score_thresh is not None else float(post_cfg.get("score_threshold_default", 0.3))
    mt = float(mask_thresh) if mask_thresh is not None else float(post_cfg.get("mask_threshold_default", 0.5))
    want_masks = task == "seg"

    out = None
    last_exc: Optional[Exception] = None
    predict_fn = getattr(model, "predict", None)
    if callable(predict_fn):
        try:
            with torch.inference_mode():
                out = predict_fn(pil, threshold=float(st))
        except TypeError:
            try:
                with torch.inference_mode():
                    out = predict_fn(pil)
            except Exception as e:
                last_exc = e
        except Exception as e:
            last_exc = e

    if out is not None:
        boxes, scores, labels, masks = parse_model_output_generic(
            out,
            img_w=int(orig_w),
            img_h=int(orig_h),
            score_thresh=float(st),
            want_masks=bool(want_masks),
            mask_thresh=float(mt),
            topk=int(topk),
        )
    else:
        tensor = T.ToTensor()(pil_in).unsqueeze(0).to(device_t)
        raw_out = None
        if module is not None:
            try:
                with torch.inference_mode():
                    raw_out = module(tensor)  # type: ignore[misc]
            except Exception as e:
                last_exc = e
        if raw_out is None:
            return InferResult(False, None, f"Inference failed: {last_exc}")

        boxes, scores, labels, masks = parse_model_output_generic(
            raw_out,
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
                    mm = unletterbox_mask(m, lb=lb, orig_w=orig_w, orig_h=orig_h, mask_thresh=float(mt))
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


class InferenceEngine:
    def __init__(
        self,
        *,
        bundle_dir: Path,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None,
        score_thresh: Optional[float] = None,
        mask_thresh: Optional[float] = None,
        checkpoint_key: Optional[str] = None,
        use_checkpoint_model: bool = False,
        strict: bool = False,
        backend: str = "auto",
        topk: Optional[int] = None,
    ):
        self.bundle_dir = bundle_dir.expanduser().resolve()
        self.cfg = load_bundle_config(self.bundle_dir)
        self.model_cfg = self.cfg.get("model_config.json", {}) or {}
        self.pre_cfg = self.cfg.get("preprocess.json", {}) or {}
        self.post_cfg = self.cfg.get("postprocess.json", {}) or {}
        classes_raw = self.cfg.get("classes.json", []) or []
        if isinstance(classes_raw, dict) and "class_names" in classes_raw:
            classes_raw = classes_raw["class_names"]
        self.class_names: List[str] = list(classes_raw) if isinstance(classes_raw, list) else []

        self.task = str(self.model_cfg.get("task") or "detect").strip().lower()
        self.topk = int(topk) if topk is not None else int(self.post_cfg.get("topk_default", 300) or 300)
        self.score_thresh = float(score_thresh) if score_thresh is not None else float(self.post_cfg.get("score_threshold_default", 0.3))
        self.mask_thresh = float(mask_thresh) if mask_thresh is not None else float(self.post_cfg.get("mask_threshold_default", 0.5))
        
        self.backend = str(backend or "auto").strip().lower()
        self.device_str = device
        self.weights_path = weights_path.expanduser().resolve() if weights_path else (self.bundle_dir / "checkpoint.pth")
        
        self.session = None
        self.model_wrapper = None
        self.model = None
        self.engine = None
        self.active_backend = None

        self._init_backend(
            checkpoint_key=checkpoint_key,
            use_checkpoint_model=use_checkpoint_model,
            strict=strict
        )

    def _init_backend(self, checkpoint_key, use_checkpoint_model, strict):
        engine_path = self.bundle_dir / "model.engine"
        onnx_path = self.bundle_dir / "model.onnx"

        # 1. Try TensorRT
        if self.backend in {"auto", "tensorrt"} and engine_path.exists():
            try:
                import tensorrt as trt
                import pycuda.autoinit
                import pycuda.driver as cuda
                
                logger = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
                if self.engine:
                    self.context = self.engine.create_execution_context()
                    self.active_backend = "tensorrt"
                    return
            except Exception as e:
                if self.backend == "tensorrt":
                    raise RuntimeError(f"Failed to load TensorRT: {e}")

        # 2. Try ONNX
        if self.backend in {"auto", "onnx"} and onnx_path.exists():
            try:
                import onnxruntime as ort
                providers = ["CPUExecutionProvider"]
                try:
                    available = set(ort.get_available_providers())
                except Exception:
                    available = set()
                wants_cuda = (self.device_str is None) or str(self.device_str).lower().startswith("cuda")
                if wants_cuda and "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                
                self.session = ort.InferenceSession(str(onnx_path), providers=providers)
                self.active_backend = "onnx"
                return
            except Exception as e:
                if self.backend == "onnx":
                    raise RuntimeError(f"Failed to load ONNX: {e}")

        # 3. Fallback to PyTorch
        try:
            import torch
            from .checkpoints import load_checkpoint_weights
            
            self.device = torch.device(self.device_str if self.device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
            model_obj = _instantiate_model(self.task, str(self.model_cfg.get("size") or "nano"), len(self.class_names) or None)
            self.model_wrapper = model_obj
            module = _unwrap_for_inference(model_obj)
            
            lr = load_checkpoint_weights(
                module,
                str(self.weights_path),
                self.device,
                checkpoint_key=checkpoint_key,
                allow_replace_model=use_checkpoint_model,
                strict=strict,
                verbose=False
            )
            if lr.replacement_model is not None:
                self.model_wrapper = lr.replacement_model
                self.model = _unwrap_for_inference(lr.replacement_model)
            else:
                self.model = module
            
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            
            self.active_backend = "pytorch"
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch backend: {e}")

    def infer(self, image: Union[Path, "Image.Image", "np.ndarray"]) -> InferResult:
        from PIL import Image
        import numpy as np

        if isinstance(image, Path):
            pil = Image.open(str(image)).convert("RGB")
            image_id = image.name
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
            image_id = "frame"
        else:
            # Assume numpy BGR
            pil = Image.fromarray(image[:, :, ::-1]).convert("RGB")
            image_id = "frame"

        orig_w, orig_h = pil.size
        policy = str(self.pre_cfg.get("resize_policy") or "letterbox").strip().lower()
        target_w = int(self.pre_cfg.get("target_w") or 640)
        target_h = int(self.pre_cfg.get("target_h") or 640)

        # Implementation of backend-specific inference using pre-loaded sessions/models
        if self.active_backend == "tensorrt":
            return self._infer_tensorrt(pil, image_id, target_w, target_h, policy, orig_w, orig_h)
        elif self.active_backend == "onnx":
            return self._infer_onnx(pil, image_id, target_w, target_h, policy, orig_w, orig_h)
        else:
            return self._infer_pytorch(pil, image_id, target_w, target_h, policy, orig_w, orig_h)

    def _infer_tensorrt(self, pil, image_id, target_w, target_h, policy, orig_w, orig_h) -> InferResult:
        import numpy as np
        import pycuda.driver as cuda
        import tensorrt as trt

        # Resolve dynamic shapes if any
        use_tensor_api = hasattr(self.engine, "num_io_tensors") and hasattr(self.context, "set_tensor_address")
        if use_tensor_api:
            input_name = self.engine.get_tensor_name(0)
            input_shape = self.engine.get_tensor_shape(input_name)
        else:
            input_shape = self.engine.get_binding_shape(0)

        if len(input_shape) >= 4:
            if input_shape[2] > 0: target_h = input_shape[2]
            if input_shape[3] > 0: target_w = input_shape[3]

        if policy == "letterbox":
            pil_in, lb = letterbox_pil(pil, target_w=target_w, target_h=target_h)
        else:
            pil_in = pil.resize((target_w, target_h))
            lb = None

        arr = np.asarray(pil_in, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
        arr = normalize_image_nchw(arr).astype(np.float32, copy=False)
        arr = np.ascontiguousarray(arr)

        # Simplified TensorRT execution for speed (reusing buffers if possible, but for now just allocate)
        # In a real production video loop, you'd pre-allocate these.
        outputs = {}
        bindings = []
        
        if use_tensor_api:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    d_input = cuda.mem_alloc(arr.nbytes)
                    cuda.memcpy_htod(d_input, arr)
                    self.context.set_tensor_address(name, int(d_input))
                else:
                    shape = self.context.get_tensor_shape(name)
                    dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                    host_out = np.empty(shape, dtype=dtype)
                    d_output = cuda.mem_alloc(host_out.nbytes)
                    self.context.set_tensor_address(name, int(d_output))
                    outputs[name] = (host_out, d_output)
            
            self.context.execute_v2([]) # Binding-less API
            for name, (host, device) in outputs.items():
                cuda.memcpy_dtoh(host, device)
            out = {n: h for n, (h, d) in outputs.items()}
        else:
            # Fallback for older TensorRT
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    d_input = cuda.mem_alloc(arr.nbytes)
                    cuda.memcpy_htod(d_input, arr)
                    bindings.append(int(d_input))
                else:
                    shape = self.engine.get_binding_shape(i)
                    dtype = trt.nptype(self.engine.get_binding_dtype(i))
                    host_out = np.empty(shape, dtype=dtype)
                    d_output = cuda.mem_alloc(host_out.nbytes)
                    bindings.append(int(d_output))
                    outputs[str(self.engine.get_binding_name(i))] = (host_out, d_output)
            
            self.context.execute_v2(bindings)
            for name, (host, device) in outputs.items():
                cuda.memcpy_dtoh(host, device)
            out = {n: h for n, (h, d) in outputs.items()}

        return self._postprocess(out, target_w, target_h, lb, orig_w, orig_h, image_id, "tensorrt")

    def _infer_onnx(self, pil, image_id, target_w, target_h, policy, orig_w, orig_h) -> InferResult:
        import numpy as np
        inputs = self.session.get_inputs()
        input_shape = inputs[0].shape
        if len(input_shape) >= 4:
            if isinstance(input_shape[2], int) and input_shape[2] > 0: target_h = input_shape[2]
            if isinstance(input_shape[3], int) and input_shape[3] > 0: target_w = input_shape[3]

        if policy == "letterbox":
            pil_in, lb = letterbox_pil(pil, target_w=target_w, target_h=target_h)
        else:
            pil_in = pil.resize((target_w, target_h))
            lb = None

        arr = np.asarray(pil_in, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
        arr = normalize_image_nchw(arr).astype(np.float32, copy=False)

        output_names = [out.name for out in self.session.get_outputs()]
        raw_outputs = self.session.run(output_names, {inputs[0].name: arr})
        out = {name: value for name, value in zip(output_names, raw_outputs)}

        return self._postprocess(out, target_w, target_h, lb, orig_w, orig_h, image_id, "onnx")

    def _infer_pytorch(self, pil, image_id, target_w, target_h, policy, orig_w, orig_h) -> InferResult:
        import torch
        import torchvision.transforms as T
        from .torch_compat import infer_backbone_patch_size
        
        patch_size = infer_backbone_patch_size(self.model)
        target_h, target_w = _adjust_dims_to_patch_size(target_h=target_h, target_w=target_w, patch_size=patch_size)

        if policy == "letterbox":
            pil_in, lb = letterbox_pil(pil, target_w=target_w, target_h=target_h)
        else:
            pil_in = pil.resize((target_w, target_h))
            lb = None

        want_masks = self.task == "seg"
        
        # Prefer the wrapper's official predict() path when available.
        predict_owner = self.model_wrapper if self.model_wrapper is not None else self.model
        predict_fn = getattr(predict_owner, "predict", None)
        if callable(predict_fn):
            try:
                with torch.inference_mode():
                    out = predict_fn(pil, threshold=self.score_thresh)
                    # predict usually handles resizing internally to original size
                    boxes, scores, labels, masks = parse_model_output_generic(
                        out, img_w=orig_w, img_h=orig_h, 
                        score_thresh=self.score_thresh, want_masks=want_masks, 
                        mask_thresh=self.mask_thresh, topk=self.topk
                    )
                    payload = detections_to_json(boxes, scores, labels, self.class_names, image_id, self.score_thresh)
                    payload["inference_backend"] = "pytorch"
                    if want_masks and masks is not None: payload["masks_present"] = True
                    return InferResult(True, payload, "ok:pytorch", boxes=boxes, scores=scores, labels=labels, masks=masks)
            except Exception:
                pass

        tensor = T.ToTensor()(pil_in).unsqueeze(0)
        tensor = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(tensor[0]).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.inference_mode():
            out = self.model(tensor)
        
        return self._postprocess(out, target_w, target_h, lb, orig_w, orig_h, image_id, "pytorch")

    def _postprocess(self, out, target_w, target_h, lb, orig_w, orig_h, image_id, backend_name) -> InferResult:
        want_masks = self.task == "seg"
        boxes, scores, labels, masks = parse_model_output_generic(
            out,
            img_w=int(target_w),
            img_h=int(target_h),
            score_thresh=self.score_thresh,
            want_masks=want_masks,
            mask_thresh=self.mask_thresh,
            topk=self.topk,
        )

        if lb is not None:
            boxes = [unletterbox_xyxy(b, lb=lb, orig_w=orig_w, orig_h=orig_h) for b in boxes]
            if want_masks and masks is not None:
                masks = [unletterbox_mask(m, lb=lb, orig_w=orig_w, orig_h=orig_h, mask_thresh=self.mask_thresh) for m in masks]

        boxes, scores, labels, masks = filter_known_class_detections(
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=self.class_names,
            masks=masks,
        )

        payload = detections_to_json(
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=self.class_names,
            image_id=image_id,
            score_thresh=self.score_thresh,
        )
        payload["inference_backend"] = backend_name
        if want_masks and masks is not None:
            payload["masks_present"] = True

        return InferResult(True, payload, f"ok:{backend_name}", boxes=boxes, scores=scores, labels=labels, masks=masks)


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
    engine = InferenceEngine(
        bundle_dir=bundle_dir,
        weights_path=weights_path,
        device=device,
        score_thresh=score_thresh,
        mask_thresh=mask_thresh,
        checkpoint_key=checkpoint_key,
        use_checkpoint_model=use_checkpoint_model,
        strict=strict,
        backend=backend,
        topk=topk
    )
    return engine.infer(image_path)
