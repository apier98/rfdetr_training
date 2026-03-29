from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .pathutil import resolve_path

from .checkpoints import load_checkpoint_weights
from .datasets import load_metadata
from .jsonutil import load_json_strict
from .model_factory import instantiate_rfdetr_model


@dataclass(frozen=True)
class ExportResult:
    ok: bool
    output_path: Optional[Path] = None
    message: str = ""


def quantize_onnx(
    *,
    onnx_path: Path,
    output_path: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    calibration_split: str = "valid",
    calibration_count: int = 100,
    height: int = 640,
    width: int = 640,
) -> ExportResult:
    """Quantize an existing ONNX model to INT8."""
    from .quantization import quantize_onnx_model

    onnx_path = resolve_path(onnx_path)
    if output_path is None:
        output_path = onnx_path.with_name(onnx_path.stem + "_quantized.onnx")
    else:
        output_path = resolve_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_images: List[Path] = []
    if dataset_dir:
        dataset_dir = resolve_path(dataset_dir)
        split_dir = dataset_dir / "coco" / calibration_split
        ann_path = split_dir / "_annotations.coco.json"
        if ann_path.exists():
            try:
                coco_data = load_json_strict(ann_path)
                images = coco_data.get("images", [])
                # Use a subset for calibration
                selected = images[: int(calibration_count)]
                for img_info in selected:
                    fname = img_info.get("file_name")
                    if fname:
                        p = split_dir / fname
                        if p.exists():
                            calibration_images.append(p)
            except Exception as e:
                print(f"Warning: Failed to load calibration data from {ann_path}: {e}")

    try:
        ok = quantize_onnx_model(
            model_path=onnx_path,
            output_path=output_path,
            calibration_data=calibration_images,
            target_h=height,
            target_w=width,
            verbose=True,
        )
        if ok:
            return ExportResult(True, output_path, f"Wrote quantized ONNX: {output_path}")
        return ExportResult(False, None, "Quantization failed (output file not created).")
    except Exception as e:
        return ExportResult(False, None, f"Quantization failed: {e}")


def _instantiate_model(task: str, size: str, num_classes: Optional[int], pretrain_weights: Optional[str] = None):
    model, _, _ = instantiate_rfdetr_model(task, size, num_classes=num_classes, pretrain_weights=pretrain_weights)
    return model


def _find_torch_module(model: object):
    import torch.nn as nn

    from .torch_compat import unwrap_torch_module

    return unwrap_torch_module(model)


def _extract_outputs(out: object, *, want_masks: bool) -> Tuple[Any, ...]:
    """Convert model forward outputs into a tuple of tensors suitable for ONNX export."""
    import torch

    def _as_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return None

    # dict-like
    if isinstance(out, dict):
        logits = out.get("pred_logits", None)
        if logits is None:
            logits = out.get("logits", None)
        boxes = out.get("pred_boxes", None)
        if boxes is None:
            boxes = out.get("boxes", None)
        masks = out.get("pred_masks", None)
        if masks is None:
            masks = out.get("masks", None)
        if masks is None:
            masks = out.get("mask", None)
        t_logits = _as_tensor(logits)
        t_boxes = _as_tensor(boxes)
        t_masks = _as_tensor(masks)
        if t_logits is not None and t_boxes is not None:
            if want_masks and t_masks is not None:
                return (t_logits, t_boxes, t_masks)
            return (t_logits, t_boxes)

    # object with attributes
    for logits_attr, boxes_attr, masks_attr in (
        ("pred_logits", "pred_boxes", "pred_masks"),
        ("logits", "boxes", "masks"),
    ):
        if hasattr(out, logits_attr) and hasattr(out, boxes_attr):
            t_logits = _as_tensor(getattr(out, logits_attr))
            t_boxes = _as_tensor(getattr(out, boxes_attr))
            if t_logits is not None and t_boxes is not None:
                if want_masks and hasattr(out, masks_attr):
                    t_masks = _as_tensor(getattr(out, masks_attr))
                    if t_masks is not None:
                        return (t_logits, t_boxes, t_masks)
                return (t_logits, t_boxes)

    # tuple/list of tensors
    if isinstance(out, (tuple, list)):
        tensors = tuple(x for x in out if _as_tensor(x) is not None)
        if len(tensors) >= (3 if want_masks else 2):
            return tensors[: (3 if want_masks else 2)]

    raise TypeError(
        "Model forward output is not exportable to ONNX (expected tensors or a dict/object with pred_logits/pred_boxes[/pred_masks]). "
        f"Got: {type(out)}"
    )


def export_onnx(
    *,
    dataset_dir: Path,
    weights: Path,
    task: str,
    size: str,
    output: Optional[Path],
    device: Optional[str],
    height: int,
    width: int,
    opset: int,
    dynamic: bool,
    use_checkpoint_model: bool,
    checkpoint_key: Optional[str],
    strict: bool,
    batchless_input: bool = False,
    half: bool = False,
) -> ExportResult:
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        return ExportResult(False, None, f"PyTorch not available: {e}")

    dataset_dir = resolve_path(dataset_dir)
    weights = resolve_path(weights)

    md = load_metadata(dataset_dir)
    class_names = md.get("class_names", []) or []
    num_classes = len(class_names) if class_names else None

    try:
        model = _instantiate_model(task, size, num_classes)
    except Exception as e:
        return ExportResult(False, None, f"Failed to instantiate model: {e}")

    # Unwrap early: `rfdetr` returns wrapper objects; the real torch module is nested.
    try:
        module = _find_torch_module(model)
    except Exception as e:
        return ExportResult(False, None, f"Failed to locate torch module inside model: {e}")

    torch_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    lr = load_checkpoint_weights(
        module,
        str(weights),
        torch_device,
        checkpoint_key=checkpoint_key,
        allow_replace_model=bool(use_checkpoint_model),
        strict=bool(strict),
        verbose=True,
    )
    if not lr.ok and lr.replacement_model is None:
        return ExportResult(False, None, f"Failed to load weights: {lr.message}")
    if lr.replacement_model is not None:
        model = lr.replacement_model
        try:
            module = _find_torch_module(model)
        except Exception as e:
            return ExportResult(False, None, f"Failed to locate torch module inside checkpoint model: {e}")

    # Ensure dummy input sizes satisfy backbone divisibility constraints (e.g. DINOv2 patch size).
    from .torch_compat import infer_backbone_patch_size

    ps = infer_backbone_patch_size(module)
    if ps:
        hh = int(height)
        ww = int(width)
        adj_h = hh - (hh % ps)
        adj_w = ww - (ww % ps)
        if adj_h <= 0 or adj_w <= 0:
            return ExportResult(False, None, f"Invalid export size {hh}x{ww} for patch_size={ps}.")
        if adj_h != hh or adj_w != ww:
            print(f"Note: adjusting export size from {hh}x{ww} to {adj_h}x{adj_w} to satisfy patch_size={ps}.")
            height, width = adj_h, adj_w
    try:
        module.to(torch_device)
    except Exception:
        pass
    module.eval()

    want_masks = (task or "").lower().strip() == "seg"

    # ONNX exporter currently does not support antialiased upsampling ops like
    # `aten::_upsample_bicubic2d_aa`. We disable `antialias=True` during export
    # to keep the graph exportable (deployment-friendly approximation).
    import contextlib
    import torch.nn.functional as F

    @contextlib.contextmanager
    def _disable_interpolate_antialias_for_onnx():
        orig = F.interpolate

        def patched(*args, **kwargs):
            # Handle both keyword and positional `antialias`.
            aa = kwargs.get("antialias", None)
            if aa is None and len(args) >= 7 and isinstance(args[6], bool):
                aa = args[6]
            if aa is True:
                if "antialias" in kwargs:
                    kwargs["antialias"] = False
                elif len(args) >= 7:
                    args = list(args)
                    args[6] = False
                    args = tuple(args)
                else:
                    kwargs["antialias"] = False
            return orig(*args, **kwargs)

        F.interpolate = patched  # type: ignore[assignment]
        try:
            yield
        finally:
            F.interpolate = orig  # type: ignore[assignment]

    @contextlib.contextmanager
    def _patch_rfdetr_projector_layernorm_for_onnx():
        """Patch rfdetr's projector.LayerNorm to use constant normalized_shape.

        The upstream implementation uses `(x.size(3),)` which creates a dynamic
        `List[int]` in the graph and fails ONNX export.
        """

        try:
            from rfdetr.models.backbone import projector as _proj  # type: ignore

            LayerNorm = getattr(_proj, "LayerNorm", None)
            if LayerNorm is None:
                yield
                return

            orig_forward = LayerNorm.forward

            def forward(self, x):
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
                x = x.permute(0, 3, 1, 2)
                return x

            LayerNorm.forward = forward
            try:
                yield
            finally:
                LayerNorm.forward = orig_forward
        except Exception:
            # If rfdetr isn't installed or the module path differs, just skip.
            yield

    @contextlib.contextmanager
    def _patch_topk_for_tensorrt_onnx():
        """Avoid exporting `TopK(axis=1)` on batch-shaped tensors for static batch=1.

        TensorRT 8.6's ONNX parser rejects the RF-DETR two-stage query-selection
        pattern when it is exported as TopK over axis 1 of a rank-2 tensor
        shaped like `(batch, proposals)`. During bundle/export we trace with a
        fixed batch size of 1, so we can safely squeeze the batch dimension for
        the TopK call and then restore it. This preserves the batch-1 semantics
        while producing a TensorRT-compatible ONNX graph.
        """

        orig_topk = torch.topk

        def patched(input, k, dim=None, largest=True, sorted=True, *, out=None):
            if (
                out is None
                and isinstance(input, torch.Tensor)
                and input.dim() == 2
                and dim in (1, -1)
            ):
                flat_input = input.reshape(-1)
                values, indices = orig_topk(
                    flat_input,
                    k,
                    dim=0,
                    largest=largest,
                    sorted=sorted,
                )
                return values.unsqueeze(0), indices.unsqueeze(0)
            return orig_topk(input, k, dim=dim, largest=largest, sorted=sorted, out=out)

        torch.topk = patched  # type: ignore[assignment]
        try:
            yield
        finally:
            torch.topk = orig_topk  # type: ignore[assignment]

    class OnnxWrapper(nn.Module):
        def __init__(self, inner: nn.Module, want_masks: bool, *, batchless_input: bool):
            super().__init__()
            self.inner = inner
            self.want_masks = want_masks
            self.batchless_input = batchless_input

        def forward(self, images: torch.Tensor):  # type: ignore[override]
            if self.batchless_input and images.dim() == 3:
                images = images.unsqueeze(0)
            out = self.inner(images)
            tensors = _extract_outputs(out, want_masks=self.want_masks)
            if self.batchless_input:
                squeezed = []
                for tensor in tensors:
                    if isinstance(tensor, torch.Tensor) and tensor.dim() > 0 and tensor.shape[0] == 1:
                        squeezed.append(tensor[0])
                    else:
                        squeezed.append(tensor)
                return tuple(squeezed)
            return tensors

    wrapper = OnnxWrapper(module, want_masks=want_masks, batchless_input=bool(batchless_input)).to(torch_device).eval()
    if half:
        wrapper.half()

    if batchless_input:
        dummy = torch.randn(3, int(height), int(width), device=torch_device)
    else:
        dummy = torch.randn(1, 3, int(height), int(width), device=torch_device)
    
    if half:
        dummy = dummy.half()

    # output paths
    if output is None:
        out_dir = dataset_dir / "exports" / "onnx"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / ("model_seg.onnx" if want_masks else "model_detect.onnx")
    else:
        output = resolve_path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["images"]
    output_names = ["pred_logits", "pred_boxes"] + (["pred_masks"] if want_masks else [])

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "images": ({1: "height", 2: "width"} if batchless_input else {0: "batch", 2: "height", 3: "width"}),
            "pred_logits": ({0: "num_queries"} if batchless_input else {0: "batch", 1: "num_queries"}),
            "pred_boxes": ({0: "num_queries"} if batchless_input else {0: "batch", 1: "num_queries"}),
        }
        if want_masks:
            dynamic_axes["pred_masks"] = (
                {0: "num_queries", 1: "mask_h", 2: "mask_w"}
                if batchless_input
                else {0: "batch", 1: "num_queries", 2: "mask_h", 3: "mask_w"}
            )

    try:
        def _do_export(opset_version: int) -> None:
            with contextlib.ExitStack() as stack:
                stack.enter_context(_disable_interpolate_antialias_for_onnx())
                stack.enter_context(_patch_rfdetr_projector_layernorm_for_onnx())
                stack.enter_context(_patch_topk_for_tensorrt_onnx())
                torch.onnx.export(
                    wrapper,
                    dummy,
                    str(output),
                    export_params=True,
                    opset_version=int(opset_version),
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                )

        _do_export(int(opset))
    except Exception as e:
        msg = str(e)
        # PyTorch sometimes only supports certain ops (e.g. antialiased bicubic resize)
        # at newer ONNX opsets. If the user asked for opset 17 (old default), retry at 18.
        if int(opset) <= 17 and "_upsample_bicubic2d_aa" in msg and "opset version 17" in msg:
            try:
                print("Note: retrying ONNX export with opset=18 (required for antialiased bicubic resize).")
                _do_export(18)
            except Exception as e2:
                return ExportResult(False, None, f"torch.onnx.export failed (after retry opset=18): {e2}")
        else:
            return ExportResult(False, None, f"torch.onnx.export failed: {e}")

    # Structural validation — catches broken graphs that torch.onnx.export
    # writes without raising (e.g. missing initialisers, bad shapes).
    try:
        import onnx  # type: ignore
        onnx.checker.check_model(str(output))
    except ImportError:
        pass  # onnx not installed; skip silently
    except Exception as val_err:
        return ExportResult(False, None, f"ONNX graph validation failed: {val_err}")

    if half:
        print(
            "Note: FP16 ONNX model requires a GPU/CUDA ONNX Runtime provider for inference. "
            "CPU execution of FP16 models is not natively supported by onnxruntime."
        )

    return ExportResult(True, output, f"Wrote ONNX: {output}")


def export_tensorrt_from_onnx(
    *,
    onnx_path: Path,
    engine_path: Optional[Path],
    height: int,
    width: int,
    fp16: bool,
    workspace_mb: int,
) -> ExportResult:
    """Build a TensorRT engine by shelling out to `trtexec` if available.

    This keeps TensorRT as an optional deployment tool (no hard dependency).
    """
    onnx_path = resolve_path(onnx_path)
    if engine_path is None:
        engine_path = onnx_path.with_suffix(".engine")
    else:
        engine_path = resolve_path(engine_path)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

    trtexec = shutil.which("trtexec")
    if not trtexec:
        return ExportResult(
            False,
            None,
            "TensorRT export requires `trtexec` on PATH. Install TensorRT and ensure `trtexec` is available.",
        )

    cmd = [
        trtexec,
        f"--onnx={str(onnx_path)}",
        f"--saveEngine={str(engine_path)}",
        f"--memPoolSize=workspace:{int(workspace_mb)}",
    ]
    if fp16:
        cmd.append("--fp16")

    try:
        proc = subprocess.run(cmd, check=False)
    except Exception as e:
        return ExportResult(False, None, f"Failed to run trtexec: {e}")

    if proc.returncode != 0:
        return ExportResult(False, None, f"trtexec failed with exit code {proc.returncode}")

    if not engine_path.exists():
        return ExportResult(False, None, f"Engine was not created: {engine_path}")

    return ExportResult(True, engine_path, f"Wrote TensorRT engine: {engine_path}")
