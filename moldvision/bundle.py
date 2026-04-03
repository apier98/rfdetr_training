from __future__ import annotations

import hashlib
import importlib.metadata
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .pathutil import resolve_path

from .datasets import load_metadata
from .export import export_onnx, export_tensorrt_from_onnx, quantize_onnx
from .jsonutil import load_json, save_json
from .model_factory import instantiate_rfdetr_model
from .torch_compat import infer_backbone_patch_size, unwrap_torch_module


@dataclass(frozen=True)
class BundleResult:
    ok: bool
    bundle_dir: Optional[Path] = None
    message: str = ""


def _safe_name(s: str) -> str:
    out = "".join([c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in (s or "")]).strip("_")
    return out or "bundle"


def _read_onnx_input_dtype(onnx_path: Path) -> Optional[str]:
    try:
        import onnx  # type: ignore
    except ImportError:
        return None

    try:
        model = onnx.load(str(onnx_path), load_external_data=False)
        if not model.graph.input:
            return None
        tensor_type = model.graph.input[0].type.tensor_type
        elem_type = int(tensor_type.elem_type)
        name = onnx.TensorProto.DataType.Name(elem_type)
        if not name:
            return None
        return str(name).strip().lower()
    except Exception:
        return None


def _default_bundle_dir(dataset_dir: Path, *, weights: Path) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    return dataset_dir / "deploy" / f"{_safe_name(weights.stem)}_{stamp}"


def _normalize_runtime_size_for_model(*, task: str, size: str, num_classes: Optional[int], height: int, width: int) -> Tuple[int, int]:
    try:
        model, _, _ = instantiate_rfdetr_model(task, size, num_classes=num_classes, pretrain_weights=None)
        module = unwrap_torch_module(model)
        patch_size = infer_backbone_patch_size(module)
    except Exception:
        return int(height), int(width)

    if not patch_size:
        return int(height), int(width)

    h = int(height)
    w = int(width)
    adj_h = h - (h % int(patch_size))
    adj_w = w - (w % int(patch_size))
    if adj_h <= 0 or adj_w <= 0:
        return h, w
    if adj_h != h or adj_w != w:
        print(f"Note: adjusting bundle runtime size from {h}x{w} to {adj_h}x{adj_w} to satisfy patch_size={patch_size}.")
    return adj_h, adj_w


def _package_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_checksums(bundle_dir: Path, *, exclude: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Return SHA-256 hex digests for every file directly in *bundle_dir*, keyed by filename."""
    skip = set(exclude or [])
    result: Dict[str, str] = {}
    for p in sorted(bundle_dir.iterdir()):
        if p.is_file() and p.name not in skip:
            result[p.name] = _sha256_file(p)
    return result


def _bundle_runtime_versions() -> Dict[str, Optional[str]]:
    return {
        "python": sys.version.split()[0],
        "torch": _package_version("torch"),
        "torchvision": _package_version("torchvision"),
        "rfdetr": _package_version("rfdetr"),
        "pillow": _package_version("pillow"),
        "numpy": _package_version("numpy"),
        "onnxruntime": (_package_version("onnxruntime") or _package_version("onnxruntime-gpu")),
    }


def _bundle_requirements_text(runtime_versions: Dict[str, Optional[str]]) -> str:
    lines = [
        "# Primary runtime requirements for this bundle.",
        "# This bundle is ONNX-first: `infer.py` will prefer model.onnx when present.",
        "",
    ]
    for dist_name, key in (
        ("onnxruntime", "onnxruntime"),
        ("pillow", "pillow"),
        ("numpy", "numpy"),
    ):
        version = runtime_versions.get(key)
        lines.append(f"{dist_name}=={version}" if version else dist_name)
    lines.append("")
    return "\n".join(lines)


def _bundle_pytorch_fallback_requirements_text(runtime_versions: Dict[str, Optional[str]]) -> str:
    lines = [
        "# Optional PyTorch fallback runtime for this bundle.",
        "# Install these only if you want `infer.py --backend pytorch` or need checkpoint-based fallback inference.",
        "# If an exact torch wheel is unavailable for your target machine, install a compatible",
        "# PyTorch build manually first, then install the remaining requirements from this file.",
        "",
    ]
    for dist_name, key in (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("rfdetr", "rfdetr"),
        ("pillow", "pillow"),
        ("numpy", "numpy"),
    ):
        version = runtime_versions.get(key)
        lines.append(f"{dist_name}=={version}" if version else dist_name)
    lines.append("")
    return "\n".join(lines)





def create_bundle(
    *,
    dataset_dir: Path,
    weights: Path,
    task: Optional[str],
    size: Optional[str],
    output_dir: Optional[Path],
    height: Optional[int],
    width: Optional[int],
    exports: Sequence[str],
    device: Optional[str],
    opset: int,
    dynamic_onnx: bool,
    use_checkpoint_model: bool,
    checkpoint_key: Optional[str],
    strict: bool,
    fp16: bool,
    workspace_mb: int,
    portable_checkpoint: bool,
    allow_raw_checkpoint_fallback: bool,
    include_raw_checkpoint: bool,
    make_zip: bool,
    make_mpk: bool = False,
    overwrite: bool = False,
    quantize: bool = False,
    calibration_split: str = "valid",
    calibration_count: int = 100,
    bundle_id: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: str = "1.0.0",
    channel: str = "stable",
    supersedes: Optional[str] = None,
    min_app_version: str = "0.0.0",
    standalone: bool = False,
) -> BundleResult:
    dataset_dir = resolve_path(dataset_dir)
    weights = resolve_path(weights)
    if not dataset_dir.exists():
        return BundleResult(False, None, f"Dataset dir not found: {dataset_dir}")
    if not weights.exists():
        return BundleResult(False, None, f"Weights not found: {weights}")

    md = load_metadata(dataset_dir)
    class_names = md.get("class_names", []) or []
    if not isinstance(class_names, list):
        class_names = []

    models_dir = dataset_dir / "models"
    trained_model_cfg = {}
    if (models_dir / "model_config.json").exists():
        try:
            trained_model_cfg = load_json(models_dir / "model_config.json")
        except Exception:
            trained_model_cfg = {}

    # Decide bundle output directory.
    bundle_dir = (resolve_path(output_dir) if output_dir else _default_bundle_dir(dataset_dir, weights=weights))
    if bundle_dir.exists():
        if not overwrite:
            return BundleResult(False, None, f"Output already exists: {bundle_dir} (use --overwrite)")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Determine preprocess target size; prefer trained resolution when present.
    trained_res = trained_model_cfg.get("resolution")
    if height is None and width is None and trained_res:
        try:
            height = int(trained_res)
            width = int(trained_res)
        except Exception:
            height = None
            width = None
    h = int(height) if height else 640
    w = int(width) if width else 640

    # Resolve task/size: prefer explicit CLI args; otherwise use training metadata.
    task_final = str((task or trained_model_cfg.get("task") or "detect")).strip().lower()
    size_final = str((size or trained_model_cfg.get("size") or "nano")).strip().lower()
    h, w = _normalize_runtime_size_for_model(
        task=task_final,
        size=size_final,
        num_classes=(int(len(class_names)) if class_names else None),
        height=int(h),
        width=int(w),
    )
    runtime_versions = _bundle_runtime_versions()

    # Write configs at bundle root.
    save_json(bundle_dir / "classes.json", list(class_names))
    model_config = {
        "format_version": 1,
        "task": task_final,
        "size": size_final,
        "num_classes": int(len(class_names)) if class_names else None,
        "class_names": list(class_names),
        "resolution": trained_res if trained_res is not None else None,
        "dataset_uuid": md.get("uuid"),
        "dataset_name": md.get("name"),
        "runtime_versions": runtime_versions,
    }
    save_json(bundle_dir / "model_config.json", model_config)

    preprocess = {
        "format_version": 1,
        "resize_policy": "square_resize",
        "target_h": int(h),
        "target_w": int(w),
        "input_color": "RGB",
        "input_layout": "NCHW",
        "input_dtype": "float32",
        "input_range": "0..1",
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "note": "Input contract: RGB -> float32 0..1 -> ImageNet mean/std normalization -> RF-DETR-style direct square resize -> model forward.",
    }
    save_json(bundle_dir / "preprocess.json", preprocess)

    # More conservative defaults for detect to avoid a wall of duplicate boxes.
    if task_final == "detect":
        score_thresh = 0.5
        topk = 100
        nms_iou = 0.3
        max_dets = 50
    else:
        score_thresh = 0.3
        topk = 300
        nms_iou = 0.7
        max_dets = 100

    postprocess = {
        "format_version": 1,
        "score_threshold_default": float(score_thresh),
        "mask_threshold_default": 0.5,
        "mask_alpha_default": 0.45,
        "topk_default": int(topk),
        "nms_iou_threshold_default": float(nms_iou),
        "mask_nms_iou_threshold_default": 0.8,
        "max_dets_default": int(max_dets),
        "min_box_size_default": 1.0,
        "note": "Postprocess: decode DETR raw outputs with RF-DETR's sigmoid + flattened top-K query/class selection, then filter degenerate boxes and apply optional NMS.",
    }
    save_json(bundle_dir / "postprocess.json", postprocess)

    requested_exports = [str(ex).strip().lower() for ex in exports if str(ex).strip()]
    if not requested_exports:
        requested_exports = ["onnx"]
    if quantize and "onnx_quantized" not in requested_exports:
        requested_exports.append("onnx_quantized")
    if "onnx_quantized" in requested_exports and "onnx" not in requested_exports:
        # We need a base ONNX to quantize, but we might not want it in the final bundle.
        # This will be handled in the export loop.
        pass
    if "tensorrt" in requested_exports and "onnx" not in requested_exports:
        requested_exports.insert(0, "onnx")

    # Write weights into bundle as the fallback/debug artifact.
    dst_weights = bundle_dir / "checkpoint.pth"
    raw_ckpt_in_bundle = None
    if include_raw_checkpoint:
        raw_ckpt_in_bundle = bundle_dir / "checkpoint_raw.pth"
        try:
            shutil.copy2(weights, raw_ckpt_in_bundle)
        except Exception:
            raw_ckpt_in_bundle = None

    checkpoint_write_mode = "copied"
    checkpoint_write_message = ""
    if portable_checkpoint:
        try:
            import torch  # type: ignore

            from .checkpoints import save_portable_checkpoint

            ok, msg = save_portable_checkpoint(
                src_path=str(weights),
                dst_path=str(dst_weights),
                device=torch.device("cpu"),
                checkpoint_key=checkpoint_key,
                verbose=False,
            )
            if ok:
                checkpoint_write_mode = "portable_state_dict"
                checkpoint_write_message = msg
            else:
                if not allow_raw_checkpoint_fallback:
                    return BundleResult(
                        False,
                        None,
                        "Portable checkpoint creation failed. Re-run with "
                        "`--allow-raw-checkpoint-fallback` if you explicitly want to copy the raw checkpoint into the bundle. "
                        f"Reason: {msg}",
                    )
                checkpoint_write_mode = "copied_fallback"
                checkpoint_write_message = msg
                shutil.copy2(weights, dst_weights)
        except Exception as e:
            if not allow_raw_checkpoint_fallback:
                return BundleResult(
                    False,
                    None,
                    "Portable checkpoint creation failed. Re-run with "
                    "`--allow-raw-checkpoint-fallback` if you explicitly want to copy the raw checkpoint into the bundle. "
                    f"Reason: {e}",
                )
            checkpoint_write_mode = "copied_fallback"
            checkpoint_write_message = str(e)
            shutil.copy2(weights, dst_weights)
    else:
        shutil.copy2(weights, dst_weights)
        checkpoint_write_mode = "raw_checkpoint"
        checkpoint_write_message = "Portable checkpoint disabled by CLI; copied checkpoint verbatim."

    # Write a self-contained inference runner (standalone mode only).
    if standalone:
        runner_path = Path(__file__).with_name("bundle_runner.py")
        if runner_path.exists():
            (bundle_dir / "infer.py").write_text(runner_path.read_text(encoding="utf-8"), encoding="utf-8")
        (bundle_dir / "requirements.txt").write_text(_bundle_requirements_text(runtime_versions), encoding="utf-8")
        (bundle_dir / "requirements-pytorch-fallback.txt").write_text(
            _bundle_pytorch_fallback_requirements_text(runtime_versions), encoding="utf-8"
        )
        src_pkg_dir = Path(__file__).resolve().parent
        dst_pkg_dir = bundle_dir / src_pkg_dir.name
        try:
            if not dst_pkg_dir.exists():
                shutil.copytree(
                    src_pkg_dir,
                    dst_pkg_dir,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                )
        except Exception:
            pass

    onnx_path: Optional[Path] = None
    quantized_path: Optional[Path] = None
    fp16_path: Optional[Path] = None
    engine_path: Optional[Path] = None
    tensorrt_status = "not_requested"
    tensorrt_message = ""
    for ex in requested_exports:
        fmt = (ex or "").strip().lower()
        if fmt == "onnx":
            res = export_onnx(
                dataset_dir=dataset_dir,
                weights=weights,
                task=task_final,
                size=size_final,
                output=(bundle_dir / "model.onnx"),
                device=device,
                height=int(h),
                width=int(w),
                opset=int(opset),
                dynamic=bool(dynamic_onnx),
                use_checkpoint_model=bool(use_checkpoint_model),
                checkpoint_key=checkpoint_key,
                strict=bool(strict),
                half=False,
            )
            if not res.ok or res.output_path is None:
                return BundleResult(False, None, f"Bundle created but ONNX export failed: {res.message}")
            onnx_path = res.output_path

        if fmt == "onnx_fp16":
            res = export_onnx(
                dataset_dir=dataset_dir,
                weights=weights,
                task=task_final,
                size=size_final,
                output=(bundle_dir / "model_fp16.onnx"),
                device=device,
                height=int(h),
                width=int(w),
                opset=int(opset),
                dynamic=bool(dynamic_onnx),
                use_checkpoint_model=bool(use_checkpoint_model),
                checkpoint_key=checkpoint_key,
                strict=bool(strict),
                half=True,
            )
            if not res.ok or res.output_path is None:
                return BundleResult(False, None, f"Bundle created but ONNX FP16 export failed: {res.message}")
            fp16_path = res.output_path

        if fmt == "onnx_quantized":
            base_onnx = onnx_path
            temp_onnx = None
            if base_onnx is None:
                temp_onnx = bundle_dir / "_tmp_model.onnx"
                res = export_onnx(
                    dataset_dir=dataset_dir,
                    weights=weights,
                    task=task_final,
                    size=size_final,
                    output=temp_onnx,
                    device=device,
                    height=int(h),
                    width=int(w),
                    opset=int(opset),
                    dynamic=bool(dynamic_onnx),
                    use_checkpoint_model=bool(use_checkpoint_model),
                    checkpoint_key=checkpoint_key,
                    strict=bool(strict),
                )
                if not res.ok or res.output_path is None:
                    return BundleResult(False, None, f"Bundle created but ONNX export (for quantization) failed: {res.message}")
                base_onnx = res.output_path

            q_res = quantize_onnx(
                onnx_path=base_onnx,
                output_path=(bundle_dir / "model_quantized.onnx"),
                dataset_dir=dataset_dir,
                calibration_split=calibration_split,
                calibration_count=calibration_count,
                height=int(h),
                width=int(w),
            )
            if temp_onnx and temp_onnx.exists():
                try:
                    temp_onnx.unlink()
                except Exception:
                    pass

            if q_res.ok:
                quantized_path = q_res.output_path
            else:
                return BundleResult(False, None, f"Bundle created but ONNX quantization failed: {q_res.message}")

        if fmt == "tensorrt":
            tensorrt_status = "requested"
            trt_onnx_path = bundle_dir / "_model_trt.onnx"
            res = export_onnx(
                dataset_dir=dataset_dir,
                weights=weights,
                task=task_final,
                size=size_final,
                output=trt_onnx_path,
                device=device,
                height=int(h),
                width=int(w),
                opset=int(opset),
                dynamic=False,
                use_checkpoint_model=bool(use_checkpoint_model),
                checkpoint_key=checkpoint_key,
                strict=bool(strict),
                batchless_input=True,
            )
            if not res.ok or res.output_path is None:
                return BundleResult(False, None, f"Bundle created but TensorRT ONNX export failed: {res.message}")
            engine_path = bundle_dir / "model.engine"
            try:
                trt_res = export_tensorrt_from_onnx(
                    onnx_path=res.output_path,
                    engine_path=engine_path,
                    height=int(h),
                    width=int(w),
                    fp16=bool(fp16),
                    workspace_mb=int(workspace_mb),
                )
            finally:
                try:
                    trt_onnx_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if not trt_res.ok:
                tensorrt_status = "failed"
                tensorrt_message = trt_res.message
                engine_path = None
                print(f"Warning: TensorRT export failed; keeping ONNX bundle as primary artifact. Reason: {trt_res.message}")
            else:
                tensorrt_status = "ok"
                tensorrt_message = trt_res.message

    preferred_onnx_path = quantized_path or fp16_path or onnx_path
    if preferred_onnx_path is not None:
        detected_input_dtype = _read_onnx_input_dtype(preferred_onnx_path)
        if detected_input_dtype:
            preprocess["input_dtype"] = detected_input_dtype
            preprocess["note"] = (
                "Input contract: RGB -> float32 0..1 -> ImageNet mean/std normalization -> RF-DETR-style direct square resize -> "
                f"cast to {detected_input_dtype} for the selected runtime artifact."
            )
            save_json(bundle_dir / "preprocess.json", preprocess)

    # Resolve bundle identity fields.
    ds_name = md.get("name") or dataset_dir.name
    resolved_model_name = model_name or ds_name or "model"
    resolved_bundle_id = bundle_id or f"{_safe_name(resolved_model_name)}-v{model_version}"

    # Inline class map: {"0": "name0", "1": "name1", ...} — required by MoldPilot.
    classes_map = {str(i): name for i, name in enumerate(class_names)}

    # Compute SHA-256 checksums for all files written so far (manifest itself excluded).
    checksums = _compute_checksums(bundle_dir, exclude=["manifest.json"])

    manifest = {
        # --- MoldPilot-required identity fields ---
        "bundle_id": resolved_bundle_id,
        "model_name": resolved_model_name,
        "model_version": model_version,
        "channel": channel,
        "supersedes": supersedes,
        "min_app_version": min_app_version,
        # Inline class map for MoldPilot OnnxInferenceService.
        "classes": classes_map,
        # Execution provider hint; MoldPilot will try providers in order.
        "runtime": {
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        },
        # SHA-256 integrity checksums for every file in the bundle.
        "checksums": checksums,
        # --- MoldVision provenance fields ---
        "format_version": 2,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset_dir": str(dataset_dir),
        "source_weights": str(weights),
        "primary_artifact": {
            "format": "onnx" if (quantized_path or fp16_path or onnx_path) else "pytorch-checkpoint",
            "path": (
                str(quantized_path.name) if quantized_path
                else (str(fp16_path.name) if fp16_path
                else (str(onnx_path.name) if onnx_path
                else str(dst_weights.name)))
            ),
            "runtime": ("onnxruntime" if (quantized_path or fp16_path or onnx_path) else "pytorch"),
            "input_dtype": (preprocess.get("input_dtype") if (quantized_path or fp16_path or onnx_path) else None),
        },
        "artifacts": {
            "onnx_path": (str(onnx_path.name) if onnx_path is not None else None),
            "onnx_fp16_path": (str(fp16_path.name) if fp16_path is not None else None),
            "onnx_quantized_path": (str(quantized_path.name) if quantized_path is not None else None),
            "tensorrt_engine_path": (str(engine_path.name) if engine_path is not None else None),
            "checkpoint_path": str(dst_weights.name),
            "raw_checkpoint_path": (str(raw_ckpt_in_bundle.name) if raw_ckpt_in_bundle else None),
        },
        "tensorrt": {
            "status": tensorrt_status,
            "message": tensorrt_message,
            "engine_path": (str(engine_path.name) if engine_path is not None else None),
        },
        "checkpoint": {
            "path": str(dst_weights.name),
            "mode": checkpoint_write_mode,
            "message": checkpoint_write_message,
            "raw_checkpoint_path": (str(raw_ckpt_in_bundle.name) if raw_ckpt_in_bundle else None),
        },
        "runtime_versions": runtime_versions,
        "standalone": standalone,
    }
    save_json(bundle_dir / "manifest.json", manifest)

    def _write_archive(suffix: str) -> None:
        archive_path = bundle_dir.with_suffix(suffix)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in bundle_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(bundle_dir)))

    if make_zip:
        _write_archive(".zip")
    if make_mpk:
        _write_archive(".mpk")

    message = f"Created bundle: {bundle_dir}"
    if tensorrt_status == "ok":
        message += " (TensorRT engine included)"
    elif tensorrt_status == "failed":
        message += f" (TensorRT export failed; ONNX remains primary: {tensorrt_message})"
    return BundleResult(True, bundle_dir, message)
