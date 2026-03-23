from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .datasets import load_metadata
from .export import export_onnx, export_tensorrt_from_onnx
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


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


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


_INFER_SCRIPT = r"""#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_bundle(bundle_dir: Path) -> Dict[str, Any]:
    bundle_dir = bundle_dir.expanduser().resolve()
    cfg = {
        "bundle_dir": str(bundle_dir),
        "model_config": _read_json(bundle_dir / "model_config.json"),
        "preprocess": _read_json(bundle_dir / "preprocess.json"),
        "postprocess": _read_json(bundle_dir / "postprocess.json"),
        "classes": _read_json(bundle_dir / "classes.json"),
    }
    return cfg


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


def instantiate_model(task: str, size: str, num_classes: Optional[int]):
    model, _, _ = instantiate_rfdetr_model(task, size, num_classes=num_classes, pretrain_weights=None)
    return model


def load_weights(model: Any, weights_path: Path, device: torch.device, *, use_checkpoint_model: bool, checkpoint_key: Optional[str], strict: bool) -> Any:
    ckpt = None
    try:
        ckpt = torch.load(str(weights_path), map_location=device)
    except Exception:
        try:
            ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

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
        # best effort wrapper
        if hasattr(model, "load"):
            model.load(str(weights_path))
            return model
        raise RuntimeError("Could not find a state_dict in checkpoint")

    # pick load target
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
    # Prefer an explicit inference method if present.
    for name in ("predict", "infer", "inference", "forward", "detect"):
        fn = getattr(model, name, None)
        if callable(fn):
            return fn(tensor)
    if callable(model):
        return model(tensor)
    raise RuntimeError("Model is not callable and has no predict/infer method")


def parse_detections(out: Any, *, model_w: int, model_h: int, score_thresh: float, topk: int) -> Tuple[List[List[float]], List[float], List[int]]:
    # decoded dict: boxes/scores/labels
    if isinstance(out, dict) and "boxes" in out:
        b = out.get("boxes")
        s = out.get("scores")
        l = out.get("labels") or out.get("classes")
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu()
            if b.numel() == 0:
                return [], [], []
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
            for i in range(int(b.shape[0])):
                sc = float(s[i].item())
                if sc < float(score_thresh):
                    continue
                boxes.append([float(x) for x in b[i].tolist()])
                scores.append(sc)
                labels.append(int(l[i].item()))
            return boxes, scores, labels

    # DETR raw dict: pred_logits/pred_boxes
    if isinstance(out, dict) and "pred_logits" in out and "pred_boxes" in out:
        logits = out["pred_logits"]
        boxes = out["pred_boxes"]
        if isinstance(logits, torch.Tensor) and logits.ndim == 3:
            logits = logits[0]
        if isinstance(boxes, torch.Tensor) and boxes.ndim == 3:
            boxes = boxes[0]
        if not isinstance(logits, torch.Tensor) or not isinstance(boxes, torch.Tensor):
            return [], [], []
        probs = logits.sigmoid()
        flat = probs.reshape(-1)
        if flat.numel() == 0:
            return [], [], []
        k = int(topk) if int(topk) > 0 else int(flat.numel())
        k = min(k, int(flat.numel()))
        scores_k, topk_indexes = torch.topk(flat, k, dim=0)
        keep = scores_k >= float(score_thresh)
        if not bool(keep.any()):
            return [], [], []
        scores_k = scores_k[keep]
        topk_indexes = topk_indexes[keep]
        labels_k = topk_indexes % logits.shape[-1]
        box_idxs = torch.div(topk_indexes, logits.shape[-1], rounding_mode="floor")
        boxes_k = boxes[box_idxs]
        order = torch.argsort(scores_k, descending=True)
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
                cx *= float(model_w); cy *= float(model_h); bw *= float(model_w); bh *= float(model_h)
            x1 = cx - bw / 2.0; y1 = cy - bh / 2.0; x2 = cx + bw / 2.0; y2 = cy + bh / 2.0
            out_boxes.append([x1, y1, x2, y2])
            out_scores.append(sc)
            out_labels.append(lab)
        return out_boxes, out_scores, out_labels

    return [], [], []


def draw_overlay(pil: Image.Image, dets: List[Dict[str, Any]]) -> Image.Image:
    img = pil.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in dets:
        b = d["bbox"]
        x1, y1, x2, y2 = [float(x) for x in b]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        label = f'{d.get("label_name", d.get("label_id"))}:{float(d.get("score", 0.0)):.2f}'
        if font is not None:
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(0, 255, 0), font=font)
        else:
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(0, 255, 0))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="Run inference using a portable RF-DETR bundle directory")
    ap.add_argument("--bundle-dir", default=".", help="Bundle directory (contains checkpoint.pth + *config.json)")
    ap.add_argument("--image", "-i", required=True, help="Path to an image")
    ap.add_argument("--weights", default=None, help="Override weights path (default: bundle/checkpoint.pth)")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, cpu (default: auto)")
    ap.add_argument("--threshold", type=float, default=None, help="Override score threshold")
    ap.add_argument("--out-json", default=None, help="Write detections JSON")
    ap.add_argument("--out-image", default=None, help="Write overlay image (PNG/JPG)")
    ap.add_argument("--use-checkpoint-model", action="store_true", help="Allow using a pickled model object from the checkpoint (trusted only)")
    ap.add_argument("--checkpoint-key", default=None, help="Explicit key containing state_dict inside checkpoint")
    ap.add_argument("--strict", action="store_true", help="Strict state_dict load (recommended for deployment correctness)")
    ap.add_argument("--topk", type=int, default=300)
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
    model = load_weights(model, weights_path, device, use_checkpoint_model=bool(args.use_checkpoint_model), checkpoint_key=args.checkpoint_key, strict=bool(args.strict))
    try:
        model.to(device).eval()
    except Exception:
        pass

    pil = Image.open(args.image).convert("RGB")
    orig = pil
    if policy == "letterbox":
        pil, lb = letterbox(pil, tw, th)
    else:
        pil = pil.resize((tw, th), resample=Image.BILINEAR)
        lb = {"orig_w": int(orig.width), "orig_h": int(orig.height), "pad_left": 0, "pad_top": 0, "scale": float(tw) / float(orig.width), "target_w": tw, "target_h": th}

    t = T.ToTensor()(pil).unsqueeze(0).to(device)

    with torch.inference_mode():
        out = run_inference(model, t)

    thresh = float(args.threshold) if args.threshold is not None else float(post_cfg.get("score_threshold_default", 0.3))
    boxes, scores, labels = parse_detections(out, model_w=int(tw), model_h=int(th), score_thresh=thresh, topk=int(args.topk))
    mapped_boxes = [unletterbox_xyxy(b, lb) for b in boxes]

    dets = []
    for b, s, l in zip(mapped_boxes, scores, labels):
        lid = int(l)
        lname = class_names[lid] if class_names and 0 <= lid < len(class_names) else str(lid)
        dets.append({"bbox": [float(x) for x in b], "score": float(s), "label_id": lid, "label_name": lname})
    payload = {"image_id": Path(args.image).name, "detections": dets}

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.out_json}")
    else:
        print(json.dumps(payload))

    if args.out_image:
        over = draw_overlay(orig, dets)
        Path(args.out_image).parent.mkdir(parents=True, exist_ok=True)
        over.save(args.out_image)
        print(f"Wrote overlay: {args.out_image}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


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
    overwrite: bool,
) -> BundleResult:
    dataset_dir = dataset_dir.expanduser().resolve()
    weights = weights.expanduser().resolve()
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
            trained_model_cfg = json.loads((models_dir / "model_config.json").read_text(encoding="utf-8"))
        except Exception:
            trained_model_cfg = {}

    # Decide bundle output directory.
    bundle_dir = (output_dir.expanduser().resolve() if output_dir else _default_bundle_dir(dataset_dir, weights=weights))
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
    _write_json(bundle_dir / "classes.json", list(class_names))
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
    _write_json(bundle_dir / "model_config.json", model_config)

    preprocess = {
        "format_version": 1,
        "resize_policy": "letterbox",
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
        "note": "Input contract: RGB -> float32 0..1 -> ImageNet mean/std normalization -> model forward.",
    }
    _write_json(bundle_dir / "preprocess.json", preprocess)

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
    _write_json(bundle_dir / "postprocess.json", postprocess)

    requested_exports = [str(ex).strip().lower() for ex in exports if str(ex).strip()]
    if not requested_exports:
        requested_exports = ["onnx"]
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

    # Write a self-contained inference runner.
    runner_path = Path(__file__).with_name("bundle_runner.py")
    (bundle_dir / "infer.py").write_text(runner_path.read_text(encoding="utf-8"), encoding="utf-8")
    (bundle_dir / "requirements.txt").write_text(_bundle_requirements_text(runtime_versions), encoding="utf-8")
    (bundle_dir / "requirements-pytorch-fallback.txt").write_text(
        _bundle_pytorch_fallback_requirements_text(runtime_versions), encoding="utf-8"
    )

    # Vendor the minimal source package into the bundle so `python infer.py` works
    # without requiring `pip install -e .` or setting PYTHONPATH.
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
        # Best-effort: bundle still contains ONNX/TensorRT exports if requested.
        pass
    (bundle_dir / "README.md").write_text(
        "\n".join(
            [
                "# Deployment Bundle",
                "",
                "This folder is intended to be copied into another project to run inference.",
                "",
                "### Usage Examples",
                "",
                "**Image Inference:**",
                "```powershell",
                "python infer.py --image path\\to\\image.jpg --out-json out.json --out-image out.png",
                "```",
                "",
                "**Video Inference:**",
                "```powershell",
                "python infer.py --video path\\to\\video.mp4 --out-video output.mp4 --display",
                "```",
                "",
                "**Webcam Inference:**",
                "```powershell",
                "python infer.py --webcam 0 --display",
                "```",
                "",
                "### Options",
                "- `--backend {auto,tensorrt,onnx,pytorch}`: Force a specific inference backend.",
                "- `--threshold 0.5`: Override the score threshold.",
                "- `--display`: Show a window with the results (useful for video/webcam).",
                "- `--out-video path`: Save the annotated video to a file.",
                "",
                "### Notes:",
                "- Primary shipped model format is ONNX (`model.onnx`). If `model.engine` is also present and TensorRT runtime is available, `infer.py` will try TensorRT first and fall back to ONNX automatically.",
                "- Preprocess keeps aspect ratio (letterbox) per `preprocess.json`.",
                "- This bundle includes a vendored copy of the `rfdetr_training` package so `infer.py` can run without installing this repo.",
                "- Install the primary runtime with `pip install -r requirements.txt`.",
                "- Install `requirements-pytorch-fallback.txt` only if you need checkpoint-based fallback inference.",
                "- TensorRT runtime is optional and environment-specific; if it is unavailable, bundle inference falls back to ONNX automatically.",
                "- `checkpoint.pth` is included as a secondary fallback/debug artifact, not the primary shipping format.",
                "- If `bundle_manifest.json` reports `checkpoint.mode=copied_fallback` or `raw_checkpoint`, the fallback checkpoint path may require unsafe checkpoint loading behavior and should be treated as trusted/debug only.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    onnx_path: Optional[Path] = None
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
            )
            if not res.ok or res.output_path is None:
                return BundleResult(False, None, f"Bundle created but ONNX export failed: {res.message}")
            onnx_path = res.output_path

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

    manifest = {
        "format_version": 2,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dataset_dir": str(dataset_dir),
        "source_weights": str(weights),
        "bundle_dir": str(bundle_dir),
        "runtime_versions": runtime_versions,
        "primary_artifact": {
            "format": "onnx" if onnx_path is not None else "pytorch-checkpoint",
            "path": (str(onnx_path.name) if onnx_path is not None else str(dst_weights.name)),
            "runtime": ("onnxruntime" if onnx_path is not None else "pytorch"),
        },
        "artifacts": {
            "onnx_path": (str(onnx_path.name) if onnx_path is not None else None),
            "tensorrt_engine_path": (str(engine_path.name) if engine_path is not None else None),
            "checkpoint_path": str(dst_weights.name),
            "raw_checkpoint_path": (str(raw_ckpt_in_bundle.name) if raw_ckpt_in_bundle else None),
        },
        "portability": {
            "portable_checkpoint_requested": bool(portable_checkpoint),
            "allow_raw_checkpoint_fallback": bool(allow_raw_checkpoint_fallback),
            "strict_checkpoint_loading": bool(strict),
            "self_contained_python_runner": True,
            "onnx_is_primary": bool(onnx_path is not None),
            "tensorrt_status": tensorrt_status,
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
        "files": sorted(({p.name for p in bundle_dir.iterdir() if p.is_file()} | {"bundle_manifest.json"})),
    }
    _write_json(bundle_dir / "bundle_manifest.json", manifest)

    if make_zip:
        zip_path = bundle_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in bundle_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(bundle_dir)))

    message = f"Created bundle: {bundle_dir}"
    if tensorrt_status == "ok":
        message += " (TensorRT engine included)"
    elif tensorrt_status == "failed":
        message += f" (TensorRT export failed; ONNX remains primary: {tensorrt_message})"
    return BundleResult(True, bundle_dir, message)
