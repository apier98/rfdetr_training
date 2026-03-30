#!/usr/bin/env python3
"""
Run live inference from a webcam using an RF-DETR model and display detections.

Now supports:
- Detection: boxes + labels
- Segmentation: boxes + labels + instance masks overlay

Examples:
  # Detection
  python scripts/infer_webcam.py --task detect --camera 0 --weights "datasets/<UUID>/models/best.pth" --size nano --threshold 0.3

  # Segmentation
  python scripts/infer_webcam.py --task seg --camera 0 --weights "datasets/<UUID>/models/best.pth" --threshold 0.3 --mask-alpha 0.45

  # ONNX backend
  python scripts/infer_webcam.py --backend onnx --onnx-model model.onnx --camera 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional, Tuple

try:
    import cv2
except Exception:
    print("OpenCV is required for webcam capture and display. Install with: pip install opencv-python")
    raise

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T


def parse_args():
    p = argparse.ArgumentParser(description="ARIA_MoldVision: Webcam inference (display only)")
    p.add_argument("--camera", "-c", type=int, default=0, help="webcam index (0,1,...)")
    p.add_argument("--weights", "-w", type=str, default=None, help="path to model weights (checkpoint .pth)")

    p.add_argument("--task", choices=["detect", "seg"], default="detect", help="detect or seg (segmentation). Default: detect")

    # Detection model size (ignored for seg)
    p.add_argument("--size", choices=["nano", "small", "base", "medium"], default="nano", help="model size to instantiate (detect only)")
    p.add_argument("--num-classes", type=int, default=None, help="override number of classes when instantiating the model (detect only)")

    p.add_argument("--threshold", type=float, default=0.3, help="score threshold for displayed detections")
    p.add_argument("--classes-file", type=str, default=None, help="path to METADATA.json or newline class-file to map class ids to names")
    p.add_argument("--device", type=str, default=None, help="torch device to use (e.g. cuda:0). Default: auto-detect")
    p.add_argument("--width", type=int, default=None, help="optional display width (will scale frame for display)")
    p.add_argument("--checkpoint-key", type=str, default=None, help="explicit key inside checkpoint that contains the state_dict")
    p.add_argument("--verbose", action="store_true", help="print debug information about checkpoint and model outputs")
    p.add_argument("--use-checkpoint-model", action="store_true", help="If checkpoint contains a pickled model object, use it")

    # Segmentation visualization
    p.add_argument("--mask-alpha", type=float, default=0.45, help="mask overlay alpha in [0..1], seg only")
    p.add_argument("--mask-thresh", type=float, default=0.5, help="threshold for mask logits/probabilities, seg only")
    # ONNX backend
    p.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch",
                   help="inference backend: pytorch (default) or onnx")
    p.add_argument("--onnx-model", type=str, default=None,
                   help="path to ONNX model file (required when --backend onnx)")
    return p.parse_args()


def load_class_names(path: str) -> List[str]:
    if not path:
        return []
    if not os.path.exists(path):
        print(f"classes file not found: {path}")
        return []
    try:
        if path.lower().endswith(".json"):
            j = json.load(open(path, "r", encoding="utf8"))
            if isinstance(j, dict) and "class_names" in j:
                return list(j["class_names"])
            if isinstance(j, list):
                return [str(x) for x in j]
        with open(path, "r", encoding="utf8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
            return lines
    except Exception as e:
        print(f"Failed to read classes file: {e}")
        return []


def instantiate_model(task: str, size: str, num_classes: Optional[int] = None):
    task = (task or "detect").lower().strip()
    if task == "seg":
        try:
            from rfdetr import RFDETRSegPreview
        except Exception:
            raise RuntimeError("Failed to import rfdetr segmentation model. Ensure `pip install rfdetr`.")
        return RFDETRSegPreview()

    # detect
    try:
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium
    except Exception:
        raise RuntimeError("Failed to import rfdetr. Install it with `pip install rfdetr` and ensure it's importable.")

    if size == "nano":
        if num_classes is not None:
            try:
                return RFDETRNano(num_classes=num_classes)
            except TypeError:
                return RFDETRNano()
        return RFDETRNano()
    if size == "small":
        if num_classes is not None:
            try:
                return RFDETRSmall(num_classes=num_classes)
            except TypeError:
                return RFDETRSmall()
        return RFDETRSmall()
    if size == "base":
        if num_classes is not None:
            try:
                return RFDETRBase(num_classes=num_classes)
            except TypeError:
                return RFDETRBase()
        return RFDETRBase()
    if size == "medium":
        if num_classes is not None:
            try:
                return RFDETRMedium(num_classes=num_classes)
            except TypeError:
                return RFDETRMedium()
        return RFDETRMedium()
    raise ValueError(size)


def try_load_weights(model, path: str, device: torch.device, checkpoint_key: str = None, verbose: bool = False, allow_replace_model: bool = False):
    if not path:
        return False, None
    if not os.path.exists(path):
        print(f"Weights file not found: {path}")
        return False, None
    print(f"Loading weights from {path}")
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        print("First attempt to load weights failed:", e)

        ckpt = None
        supports_weights_only = False
        try:
            import inspect as _inspect
            supports_weights_only = "weights_only" in _inspect.signature(torch.load).parameters
        except Exception:
            supports_weights_only = False

        if supports_weights_only:
            try:
                import argparse as _argparse
                try:
                    from torch.serialization import safe_globals as _safe_globals  # type: ignore
                except Exception:
                    _safe_globals = None
                if _safe_globals is not None:
                    if verbose:
                        print("Retrying torch.load with weights_only=True and safe_globals([argparse.Namespace])...")
                    with _safe_globals([_argparse.Namespace]):
                        ckpt = torch.load(path, map_location=device, weights_only=True)
            except Exception as e_safe:
                if verbose:
                    print("Safe weights_only load failed:", e_safe)

        if ckpt is None:
            print("Retrying torch.load with weights_only=False (this may execute code from the checkpoint).")
        try:
            if supports_weights_only:
                ckpt = torch.load(path, map_location=device, weights_only=False)
            else:
                ckpt = torch.load(path, map_location=device)
        except Exception as e2:
            print("Fallback load also failed:", e2)
            return False, None

    if verbose:
        try:
            print(f"Loaded checkpoint object of type {type(ckpt)}")
            if isinstance(ckpt, dict):
                print("Top-level keys:", list(ckpt.keys()))
        except Exception:
            pass

    # If user explicitly allows it, prefer returning a pickled model object from the checkpoint.
    if allow_replace_model and isinstance(ckpt, dict):
        for candidate in ("ema_model", "model"):
            if candidate not in ckpt:
                continue
            maybe = ckpt.get(candidate)
            try:
                import torch.nn as nn

                if isinstance(maybe, nn.Module):
                    if verbose:
                        print(f"Using checkpoint['{candidate}'] as replacement model ({type(maybe)}).")
                    return True, maybe
            except Exception:
                pass
            if hasattr(maybe, "state_dict") and callable(getattr(maybe, "state_dict")):
                if verbose:
                    print(f"Using checkpoint['{candidate}'] as replacement model ({type(maybe)}).")
                return True, maybe

    state = None
    if isinstance(ckpt, dict):
        if checkpoint_key and checkpoint_key in ckpt:
            state = ckpt[checkpoint_key]
        else:
            for key in ("model_state_dict", "state_dict", "model", "net"):
                if key in ckpt:
                    state = ckpt[key]
                    break
        if state is None:
            if any(k.startswith("backbone") or k.startswith("transformer") or k.startswith("class_embed") for k in ckpt.keys()):
                state = ckpt
        if state is None:
            for candidate in ("model", "ema_model"):
                if candidate in ckpt:
                    try:
                        maybe = ckpt[candidate]
                        if hasattr(maybe, "state_dict") and callable(getattr(maybe, "state_dict")):
                            if verbose:
                                print(f"Extracting state_dict() from checkpoint['{candidate}'] object of type {type(maybe)}")
                            state = maybe.state_dict()
                            break
                        if isinstance(maybe, dict):
                            state = maybe
                            break
                    except Exception as e:
                        if verbose:
                            print(f"Could not extract state_dict from checkpoint['{candidate}']: {e}")

    if state is None:
        if allow_replace_model and isinstance(ckpt, dict):
            for candidate in ("ema_model", "model"):
                if candidate in ckpt:
                    maybe = ckpt[candidate]
                    if verbose:
                        print(f"Found checkpoint['{candidate}'] object of type {type(maybe)}; returning it as replacement model")
                    return True, maybe

        print("Could not autodetect a state_dict in the checkpoint. Trying to load raw checkpoint into the model if supported.")
        try:
            if hasattr(model, "load"):
                model.load(path)
                return True, None
        except Exception:
            pass
        return False, None

    # find target to load
    def find_load_target(obj, max_depth=3):
        import torch.nn as nn
        if hasattr(obj, "load_state_dict") and callable(getattr(obj, "load_state_dict")):
            return obj
        if isinstance(obj, nn.Module):
            return obj
        for attr in ("model", "net", "network", "module", "detector", "backbone", "transformer"):
            maybe = getattr(obj, attr, None)
            if maybe is not None:
                if hasattr(maybe, "load_state_dict") and callable(getattr(maybe, "load_state_dict")):
                    return maybe
                if isinstance(maybe, nn.Module):
                    return maybe
        if max_depth <= 0:
            return None
        try:
            for name, val in list(vars(obj).items()):
                if name.startswith("_"):
                    continue
                if val is None:
                    continue
                if hasattr(val, "load_state_dict") and callable(getattr(val, "load_state_dict")):
                    return val
                if isinstance(val, nn.Module):
                    return val
                if hasattr(val, "__dict__"):
                    found = find_load_target(val, max_depth=max_depth - 1)
                    if found is not None:
                        return found
        except Exception:
            pass
        return None

    target = find_load_target(model) or model

    if hasattr(target, "load_state_dict"):
        try:
            # filter by shape match to avoid head mismatch explosions
            try:
                target_state = target.state_dict()
                filtered = {k: v for k, v in state.items() if k in target_state and getattr(v, "shape", None) == getattr(target_state[k], "shape", None)}
            except Exception:
                filtered = state
            target.load_state_dict(filtered, strict=False)
            try:
                if hasattr(target, "to") and callable(getattr(target, "to")):
                    target.to(device)
            except Exception:
                pass
            return True, None
        except Exception as e:
            print(f"load_state_dict failed: {e}")
            try:
                fixed = {(k.replace("module.", "")): v for k, v in state.items()}
                target.load_state_dict(fixed, strict=False)
                return True, None
            except Exception as e2:
                print(f"Second load attempt failed: {e2}")
                return False, None

    # wrappers
    for name in ("load_state_dict", "load_weights", "load", "load_from_checkpoint"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                fn(path)
                return True, None
            except Exception as e:
                print(f"Tried wrapper.{name} but it failed: {e}")

    return False, None


def preprocess_frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    # Ensure frame is HxWx3 BGR
    try:
        if frame is None:
            raise ValueError("frame is None")
        # Grayscale -> BGR
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # BGRA -> BGR
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    except Exception:
        # Best-effort fallback: stack channels if unexpected shape
        try:
            if frame is not None and frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=2)
        except Exception:
            pass

    # Normalize dtype to uint8 so PIL handles it predictably
    try:
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype("uint8")
            else:
                frame = frame.astype("uint8")
    except Exception:
        pass

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    transform = T.Compose([T.ToTensor()])
    t = transform(pil).unsqueeze(0).to(device)
    return t


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    try:
        t = tensor.detach().cpu().squeeze(0)
        t = T.ToPILImage()(t)
        return t
    except Exception:
        arr = (tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(arr)


def run_inference(model, module_for_torch, tensor: torch.Tensor):
    """Try several inference call patterns and return raw output."""
    last_exc = None

    def _looks_empty(o) -> bool:
        if o is None:
            return True
        try:
            empty_attr = getattr(o, "empty", None)
            if isinstance(empty_attr, (bool, np.bool_)):
                return bool(empty_attr)
            if callable(empty_attr):
                try:
                    return bool(empty_attr())
                except Exception:
                    pass
        except Exception:
            pass

        # supervision.Detections commonly exposes `.xyxy`
        for attr in ("xyxy", "boxes", "pred_boxes"):
            if not hasattr(o, attr):
                continue
            try:
                v = getattr(o, attr)
                if isinstance(v, torch.Tensor):
                    return v.numel() == 0 or (v.ndim >= 1 and v.shape[0] == 0)
                arr = np.asarray(v)
                return arr.size == 0 or (arr.ndim >= 1 and arr.shape[0] == 0)
            except Exception:
                continue

        if isinstance(o, dict):
            for k in ("boxes", "pred_boxes"):
                if k not in o:
                    continue
                try:
                    v = o[k]
                    if isinstance(v, torch.Tensor):
                        return v.numel() == 0 or (v.ndim >= 1 and v.shape[0] == 0)
                    arr = np.asarray(v)
                    return arr.size == 0 or (arr.ndim >= 1 and arr.shape[0] == 0)
                except Exception:
                    continue

        return False

    if module_for_torch is not None and callable(module_for_torch):
        try:
            out = module_for_torch(tensor)
            if not _looks_empty(out):
                return out
        except Exception as e:
            last_exc = e

        # try passing a PIL image if the module expects PIL/numpy OR tensor gave empty outputs
        try:
            pil = tensor_to_pil(tensor)
            out = module_for_torch(pil)
            return out
        except Exception as e2:
            last_exc = e2

    candidates = ["predict", "infer", "inference", "forward", "detect"]
    for name in candidates:
        fn = getattr(model, name, None)
        if not callable(fn):
            continue
        try:
            out = fn(tensor)
            if not _looks_empty(out):
                return out
        except Exception as e:
            last_exc = e

        # try PIL fallback for models that expect PIL/numpy inputs OR tensor gave empty outputs
        try:
            pil = tensor_to_pil(tensor)
            out = fn(pil)
            return out
        except Exception as e2:
            last_exc = e2
            continue

    if callable(model):
        try:
            return model(tensor)
        except Exception as e:
            last_exc = e

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Could not find a callable inference method on the model")


def _normalize_masks(masks, mask_thresh: float) -> Optional[List[np.ndarray]]:
    if masks is None:
        return None
    if isinstance(masks, list):
        out = []
        for m in masks:
            mm = m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
            if mm.ndim == 3 and mm.shape[0] == 1:
                mm = mm[0]
            if mm.ndim != 2:
                continue
            out.append(mm > mask_thresh)
        return out if out else None

    mm = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.asarray(masks)

    if mm.ndim == 2:
        return [mm > mask_thresh]
    if mm.ndim == 3:
        return [(mm[i] > mask_thresh) for i in range(mm.shape[0])]
    if mm.ndim == 4 and mm.shape[1] == 1:
        return [(mm[i, 0] > mask_thresh) for i in range(mm.shape[0])]
    return None


def parse_detections(output, img_w: int, img_h: int, score_thresh: float, want_masks: bool, mask_thresh: float):
    """Return boxes, scores, labels, masks (masks optional)."""
    boxes: List[List[float]] = []
    scores: List[float] = []
    labels: List[int] = []
    masks: Optional[List[np.ndarray]] = None

    # supervision-like
    try:
        modname = output.__class__.__module__ if hasattr(output, "__class__") else ""
    except Exception:
        modname = ""
    if modname.startswith("supervision") or "Detections" in getattr(output, "__class__", type(output)).__name__:
        try:
            b = None
            if hasattr(output, "xyxy"):
                b = np.asarray(getattr(output, "xyxy"))
            elif hasattr(output, "xywh"):
                tmp = np.asarray(getattr(output, "xywh"))
                if tmp.size:
                    cx = tmp[:, 0]; cy = tmp[:, 1]; w_ = tmp[:, 2]; h_ = tmp[:, 3]
                    x1 = cx - w_ / 2; y1 = cy - h_ / 2; x2 = cx + w_ / 2; y2 = cy + h_ / 2
                    b = np.stack([x1, y1, x2, y2], axis=1)

            s = None
            for cand in ("confidence", "confidences", "scores", "score"):
                if hasattr(output, cand):
                    s = np.asarray(getattr(output, cand))
                    break

            l = None
            for cand in ("class_id", "class_ids", "labels", "category_id", "class"):
                if hasattr(output, cand):
                    l = np.asarray(getattr(output, cand))
                    break

            if want_masks:
                m = None
                for cand in ("mask", "masks", "segmentation", "segmentations"):
                    if hasattr(output, cand):
                        m = getattr(output, cand)
                        break
                masks = _normalize_masks(m, mask_thresh)

            if b is not None and b.size:
                keep_idx = []
                for i in range(b.shape[0]):
                    bx = b[i]
                    if float(bx.max()) <= 1.0:
                        bx_pixels = [float(bx[0] * img_w), float(bx[1] * img_h), float(bx[2] * img_w), float(bx[3] * img_h)]
                    else:
                        bx_pixels = [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])]
                    if bx_pixels[2] < bx_pixels[0] or bx_pixels[3] < bx_pixels[1]:
                        cx, cy, w_, h_ = bx_pixels
                        bx_pixels = [cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2]

                    sc = float(s[i]) if s is not None and i < len(s) else 1.0
                    if sc < score_thresh:
                        continue

                    keep_idx.append(i)
                    boxes.append(bx_pixels)
                    scores.append(sc)
                    lab = int(l[i]) if l is not None and i < len(l) else 0
                    labels.append(int(lab))

                if want_masks and masks is not None:
                    masks = [masks[i] for i in keep_idx if i < len(masks)]

                return boxes, scores, labels, masks
        except Exception:
            pass

    # dict/tensor
    try:
        out = output[0] if isinstance(output, (list, tuple)) else output

        if isinstance(out, dict):
            if want_masks:
                m = None
                for k in ("masks", "mask", "pred_masks", "segmentation"):
                    if k in out:
                        m = out[k]
                        break
                masks = _normalize_masks(m, mask_thresh)

            if "boxes" in out:
                b = out["boxes"]
                s = out.get("scores", None)
                l = out.get("labels", out.get("classes", None))
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

                    keep = []
                    for i in range(b.shape[0]):
                        if float(s[i].item()) < score_thresh:
                            continue
                        keep.append(i)
                        boxes.append([float(x) for x in b[i].tolist()])
                        scores.append(float(s[i].item()))
                        labels.append(int(l[i].item()))

                    if want_masks and masks is not None:
                        masks = [masks[i] for i in keep if i < len(masks)]

                    return boxes, scores, labels, masks

        if isinstance(output, torch.Tensor):
            t = output.detach().cpu()
            if t.ndim == 2 and t.shape[1] >= 6:
                for i in range(t.shape[0]):
                    sc = float(t[i, 4].item())
                    if sc < score_thresh:
                        continue
                    boxes.append([float(x) for x in t[i, 0:4].tolist()])
                    scores.append(sc)
                    labels.append(int(t[i, 5].item()))
                return boxes, scores, labels, None
    except Exception:
        pass

    return [], [], [], None


def _color_for_id(i: int) -> Tuple[int, int, int]:
    # Deterministic-ish “random” color per label/id
    r = (37 * (i + 1)) % 255
    g = (17 * (i + 1)) % 255
    b = (97 * (i + 1)) % 255
    return int(b), int(g), int(r)  # BGR for OpenCV


def overlay_masks(frame_bgr: np.ndarray, masks: List[np.ndarray], labels: List[int], alpha: float = 0.45):
    """Alpha-blend boolean masks onto frame (in-place)."""
    if not masks:
        return

    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    for i, m in enumerate(masks):
        if m is None:
            continue
        mm = m
        # ensure HxW
        if mm.shape[0] != h or mm.shape[1] != w:
            # resize mask to frame size (nearest)
            mm_u8 = (mm.astype(np.uint8) * 255)
            mm_u8 = cv2.resize(mm_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            mm = mm_u8 > 127

        color = _color_for_id(labels[i] if i < len(labels) else i)
        overlay[mm] = color

    cv2.addWeighted(overlay, float(alpha), frame_bgr, float(1.0 - alpha), 0, dst=frame_bgr)


def draw_detections(frame: np.ndarray, boxes: List[List[float]], scores: List[float], labels: List[int], class_names: List[str]):
    for box, score, lab in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(max(0, round(v))) for v in box]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name = str(lab)
        if class_names and 0 <= lab < len(class_names):
            name = class_names[lab]
        text = f"{name}: {score:.2f}"
        cv2.putText(frame, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def load_onnx_session(onnx_path: str, device_str: Optional[str]):
    """Create an onnxruntime InferenceSession, preferring CUDA if available."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime-gpu")
    providers = ["CPUExecutionProvider"]
    try:
        available = set(ort.get_available_providers())
    except Exception:
        available = set()
    wants_cuda = device_str is None or str(device_str).lower().startswith("cuda")
    if wants_cuda and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"ONNX Runtime providers in use: {session.get_providers()}")
    # Warn when loading an fp16 model without TensorRT: onnxruntime's CUDA provider
    # lacks fp16 kernels for some ops (Sqrt, Tile), causing Memcpy roundtrips that
    # are slower than fp32 on CUDA. Use model.onnx (fp32) for onnxruntime inference;
    # fp16 ONNX is designed as an intermediate for TensorRT export.
    try:
        inp_type = str(session.get_inputs()[0].type).lower()
        using_cuda = any("cuda" in p.lower() for p in session.get_providers())
        using_trt = any("tensorrt" in p.lower() for p in session.get_providers())
        if ("float16" in inp_type or "fp16" in inp_type) and using_cuda and not using_trt:
            print(
                "Warning: fp16 ONNX model loaded with CUDAExecutionProvider but without TensorRT. "
                "Some ops (Sqrt, Tile) lack fp16 CUDA kernels and will fall back to CPU with "
                "Memcpy overhead, making fp16 slower than fp32. "
                "Use model.onnx (fp32) for onnxruntime inference, or export a TensorRT engine for fp16."
            )
    except Exception:
        pass
    return session


def onnx_input_hw(session, default_h: int = 560, default_w: int = 560) -> Tuple[int, int]:
    """Return (target_h, target_w) from the session's first input shape."""
    target_h, target_w = default_h, default_w
    try:
        shape = session.get_inputs()[0].shape
        if len(shape) >= 4:
            if isinstance(shape[2], int) and shape[2] > 0:
                target_h = int(shape[2])
            if isinstance(shape[3], int) and shape[3] > 0:
                target_w = int(shape[3])
    except Exception:
        pass
    return target_h, target_w


def preprocess_frame_for_onnx(frame_bgr: np.ndarray, target_h: int, target_w: int):
    """BGR frame → (NCHW float32 array, Letterbox, orig_h, orig_w)."""
    from moldvision.postprocess import letterbox_pil, normalize_image_nchw
    if frame_bgr.ndim == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    orig_w, orig_h = pil.width, pil.height
    pil_lb, lb = letterbox_pil(pil, target_w=target_w, target_h=target_h)
    arr = np.asarray(pil_lb, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    arr = np.asarray(normalize_image_nchw(arr), dtype=np.float32)
    return arr, lb, orig_h, orig_w


def run_onnx_frame(
    session,
    frame_bgr: np.ndarray,
    target_h: int,
    target_w: int,
    score_thresh: float,
    want_masks: bool,
    mask_thresh: float,
) -> Tuple[List[List[float]], List[float], List[int], Optional[List[np.ndarray]]]:
    """Run ONNX inference on one BGR frame. Returns (boxes, scores, labels, masks)."""
    from moldvision.postprocess import parse_model_output_generic, unletterbox_xyxy, unletterbox_mask
    arr, lb, orig_h, orig_w = preprocess_frame_for_onnx(frame_bgr, target_h, target_w)
    # handle fp16 models
    try:
        t = str(session.get_inputs()[0].type).lower()
        if "float16" in t or "fp16" in t:
            arr = arr.astype(np.float16, copy=False)
    except Exception:
        pass
    try:
        inp = session.get_inputs()[0]
        output_names = [o.name for o in session.get_outputs()]
        raw = session.run(output_names, {inp.name: arr})
        out = {name: val for name, val in zip(output_names, raw)}
    except Exception as e:
        print(f"ONNX inference failed: {e}")
        return [], [], [], None
    boxes, scores, labels, masks = parse_model_output_generic(
        out,
        img_w=target_w,
        img_h=target_h,
        score_thresh=score_thresh,
        want_masks=want_masks,
        mask_thresh=mask_thresh,
    )
    boxes = [unletterbox_xyxy(b, lb=lb, orig_w=orig_w, orig_h=orig_h) for b in boxes]
    if want_masks and masks:
        masks = [unletterbox_mask(m, lb=lb, orig_w=orig_w, orig_h=orig_h, mask_thresh=mask_thresh) for m in masks]
    return boxes, scores, labels, masks


def main():
    args = parse_args()

    if args.backend == "onnx" and not args.onnx_model:
        print("Error: --onnx-model is required when --backend onnx")
        return

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    class_names = load_class_names(args.classes_file) if args.classes_file else []

    if args.backend == "onnx":
        session = load_onnx_session(args.onnx_model, args.device)
        target_h, target_w = onnx_input_hw(session)

        def run_frame_fn(frame_bgr):
            return run_onnx_frame(
                session, frame_bgr, target_h, target_w,
                args.threshold, args.task == "seg", args.mask_thresh,
            )
    else:
        if args.task == "seg" and args.size != "nano":
            print("Note: --size is ignored for --task seg")

        print("Instantiating model...")
        model = instantiate_model(args.task, args.size, args.num_classes)

        if args.weights:
            ok, replacement = try_load_weights(
                model, args.weights, device,
                checkpoint_key=args.checkpoint_key,
                verbose=args.verbose,
                allow_replace_model=args.use_checkpoint_model
            )
            if not ok:
                print("Warning: could not load weights. Inference may fail or use random weights.")
            if replacement is not None:
                if args.verbose:
                    print(f"Replacing instantiated model with checkpoint object of type {type(replacement)}")
                model = replacement

        module_for_torch = None
        if hasattr(model, "to") and callable(getattr(model, "to")):
            module_for_torch = model
        else:
            for attr in ("model", "net", "network", "module", "detector", "backbone"):
                inner = getattr(model, attr, None)
                if inner is not None and hasattr(inner, "to") and callable(getattr(inner, "to")):
                    module_for_torch = inner
                    if args.verbose:
                        print(f"Using inner module '{attr}' for device placement")
                    break

        if module_for_torch is not None:
            try:
                module_for_torch.to(device)
            except Exception as e:
                print(f"Warning: moving module to device failed: {e}. Falling back to CPU.")
                device = torch.device("cpu")
            try:
                module_for_torch.eval()
            except Exception:
                pass
        else:
            if args.verbose:
                print("Warning: could not find an inner torch module to move to device; using CPU.")
            device = torch.device("cpu")

        if hasattr(model, "to") and callable(getattr(model, "to")):
            try:
                model.to(device)
            except Exception:
                pass

        def run_frame_fn(frame_bgr):
            h, w = frame_bgr.shape[:2]
            tensor = preprocess_frame_to_tensor(frame_bgr, device)
            with torch.no_grad():
                try:
                    out = run_inference(model, module_for_torch, tensor)
                except Exception as e:
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    print(f"Model forward failed: {e}")
                    out = None

            if out is None:
                return [], [], [], None

            if args.verbose:
                try:
                    if isinstance(out, dict):
                        print("Output keys:", list(out.keys()))
                    elif isinstance(out, torch.Tensor):
                        print("Tensor shape:", out.shape)
                    else:
                        print("Output type:", type(out))
                except Exception:
                    pass

            return parse_detections(
                out, w, h,
                score_thresh=args.threshold,
                want_masks=(args.task == "seg"),
                mask_thresh=args.mask_thresh,
            )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Unable to open camera index {args.camera}")
        return

    fps = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            start = time.time()
            boxes, scores, labels, masks = run_frame_fn(frame)

            disp = frame.copy()
            if args.task == "seg" and masks:
                overlay_masks(disp, masks, labels, alpha=args.mask_alpha)
            draw_detections(disp, boxes, scores, labels, class_names)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - start)) if now - start > 0 else fps
            cv2.putText(disp, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            if args.width:
                scale = args.width / disp.shape[1]
                disp = cv2.resize(disp, (args.width, int(disp.shape[0] * scale)))

            cv2.imshow("ARIA_MoldVision (press q to quit)", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
