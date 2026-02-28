"""Inference helpers: robust checkpoint loader and JSON conversion utilities.

Designed to be copy/pasted into another project. Provides:
- instantiate_model(size, num_classes=None, task="detect")
- detect_num_classes_from_checkpoint(path, checkpoint_key=None)
- load_checkpoint_weights(model, path, device, checkpoint_key=None, allow_replace_model=False, verbose=False)
- parse_model_output(output, img_w, img_h, score_thresh, return_masks=False, mask_thresh=0.5)
- detections_to_json(boxes, scores, labels, class_names=None, image_id=None, score_thresh=0.3)

Keep this as a lightweight dependency: only requires torch and numpy; `rfdetr` is optional when instantiating models.
"""

from __future__ import annotations

from typing import Tuple, List, Optional, Any, Union
import os
import torch
import numpy as np


def instantiate_model(size: str, num_classes: Optional[int] = None, task: str = "detect"):
    """Instantiate RFDETR model by name.

    task:
      - "detect": RFDETRNano/Small/Base/Medium
      - "seg":    RFDETRSegPreview (size ignored)

    Returns an object (may be wrapper) or raises on import error.
    """
    task = (task or "detect").lower().strip()
    if task not in ("detect", "seg"):
        raise ValueError(f"Unknown task: {task}")

    try:
        if task == "seg":
            from rfdetr import RFDETRSegPreview
            return RFDETRSegPreview()
        else:
            from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium
    except Exception:
        raise RuntimeError("Failed to import rfdetr. Install it with `pip install rfdetr` and ensure it's importable.")

    def _ctor(cls):
        # Many rfdetr versions will download/load pretrained weights if pretrain_weights is omitted.
        # Pass pretrain_weights=None as best-effort to keep deployment/debug inference deterministic.
        kwargs = {"pretrain_weights": None}
        if num_classes is not None:
            kwargs["num_classes"] = int(num_classes)
        try:
            return cls(**kwargs)
        except TypeError:
            # older versions may not accept pretrain_weights and/or num_classes
            try:
                kwargs.pop("pretrain_weights", None)
                return cls(**kwargs)
            except TypeError:
                return cls()

    if size == "nano":
        return _ctor(RFDETRNano)
    if size == "small":
        return _ctor(RFDETRSmall)
    if size == "base":
        return _ctor(RFDETRBase)
    if size == "medium":
        return _ctor(RFDETRMedium)
    raise ValueError(f"Unknown model size: {size}")


def detect_num_classes_from_checkpoint(path: str, checkpoint_key: Optional[str] = None, verbose: bool = False) -> Optional[int]:
    if not path or not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return None

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
            if any(("transformer" in k or "class_embed" in k or "enc_out_class_embed" in k) for k in ckpt.keys()):
                state = ckpt
    if not isinstance(state, dict):
        return None

    candidates = []
    for k, v in state.items():
        if "class_embed" in k or "enc_out_class_embed" in k or k.endswith(".class_embed"):
            if hasattr(v, "shape") and len(getattr(v, "shape", [])) >= 1:
                candidates.append(int(v.shape[0]))
    if len(candidates) == 0:
        return None
    try:
        from collections import Counter
        c = Counter(candidates)
        most, _ = c.most_common(1)[0]
        return int(most)
    except Exception:
        return int(max(candidates))


def find_load_target(obj, max_depth=3):
    """Find an inner object that supports load_state_dict or is an nn.Module."""
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


def _set_module_by_path(root, path: str, new_mod):
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_mod
    else:
        setattr(parent, last, new_mod)


def load_checkpoint_weights(
    model,
    path: str,
    device: torch.device,
    checkpoint_key: Optional[str] = None,
    allow_replace_model: bool = False,
    verbose: bool = False,
) -> Tuple[bool, Optional[object]]:
    """Load checkpoint into `model` (or return a replacement model if checkpoint contains a pickled model and allow_replace_model=True).

    Returns (success: bool, replacement_model_or_None).
    """
    if not path:
        return False, None
    if not os.path.exists(path):
        if verbose:
            print(f"Weights file not found: {path}")
        return False, None
    if verbose:
        print(f"Loading weights from {path}")
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        if verbose:
            print("First attempt to load weights failed:", e)

        ckpt = None
        supports_weights_only = False
        try:
            import inspect as _inspect

            supports_weights_only = "weights_only" in _inspect.signature(torch.load).parameters
        except Exception:
            supports_weights_only = False

        # PyTorch 2.6 defaults torch.load(weights_only=True), which can fail if the checkpoint
        # contains non-tensor metadata (e.g. argparse.Namespace). Try a safer allowlist first.
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
            if verbose:
                print("Retrying torch.load with weights_only=False (this may execute code from the checkpoint).")
            try:
                if supports_weights_only:
                    ckpt = torch.load(path, map_location=device, weights_only=False)
                else:
                    ckpt = torch.load(path, map_location=device)
            except Exception as e2:
                if verbose:
                    print("Fallback load also failed:", e2)
                return False, None

    # If user explicitly allows it, prefer returning a pickled model object from the checkpoint.
    # This can preserve the exact architecture/config (e.g., patch size / positional encodings)
    # and avoid subtle mismatches when instantiating a "default" model class.
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
            if any(k.startswith("backbone") or k.startswith("transformer") or "class_embed" in k for k in ckpt.keys()):
                state = ckpt
        if state is None:
            for candidate in ("model", "ema_model"):
                if candidate in ckpt:
                    maybe = ckpt[candidate]
                    if hasattr(maybe, "state_dict") and callable(getattr(maybe, "state_dict")):
                        if verbose:
                            print(f"Extracting state_dict() from checkpoint['{candidate}'] object of type {type(maybe)}")
                        state = maybe.state_dict()
                        break
                    if isinstance(maybe, dict):
                        state = maybe
                        break

    if state is None:
        if allow_replace_model and isinstance(ckpt, dict):
            for candidate in ("ema_model", "model"):
                if candidate in ckpt:
                    maybe = ckpt[candidate]
                    if verbose:
                        print(f"Found checkpoint['{candidate}'] object of type {type(maybe)}; returning it as replacement model")
                    return True, maybe

        if verbose:
            print("Could not autodetect a state_dict in the checkpoint. Trying wrapper load if available.")
        try:
            if hasattr(model, "load"):
                model.load(path)
                return True, None
        except Exception:
            pass
        return False, None

    target = find_load_target(model)
    if target is None:
        target = model
        if not hasattr(target, "load_state_dict"):
            for attr in ("model", "net", "network", "module", "detector", "backbone"):
                inner = getattr(model, attr, None)
                if inner is not None and hasattr(inner, "load_state_dict"):
                    if verbose:
                        print(f"Found inner module '{attr}' - loading state into it")
                    target = inner
                    break

    if not hasattr(target, "load_state_dict"):
        for name in ("load_state_dict", "load_weights", "load", "load_from_checkpoint"):
            fn = getattr(model, name, None)
            if callable(fn):
                try:
                    fn(path)
                    return True, None
                except Exception as e:
                    if verbose:
                        print(f"Tried wrapper.{name} but it failed: {e}")
        return False, None

    try:
        try:
            target_state = target.state_dict()
            filtered = {}
            skipped = []
            for k, v in state.items():
                if k in target_state:
                    try:
                        if getattr(v, "shape", None) == getattr(target_state[k], "shape", None):
                            filtered[k] = v
                        else:
                            skipped.append((k, getattr(v, "shape", None), getattr(target_state[k], "shape", None)))
                    except Exception:
                        skipped.append((k, None, getattr(target_state[k], "shape", None)))
            if verbose:
                print(f"Loading {len(filtered)} matched tensors from checkpoint; skipping {len(skipped)} mismatched tensors")

            # try replace classifier heads when mismatch
            try:
                import torch.nn as nn
                replaced = []
                for k, ck_shape, tgt_shape in skipped:
                    if ("class_embed" in k or "enc_out_class_embed" in k) and ck_shape is not None and tgt_shape is not None:
                        mod_path = k.rsplit(".", 1)[0]
                        try:
                            parent = target
                            for part in mod_path.split("."):
                                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
                            existing = parent
                        except Exception:
                            existing = None
                        if existing is not None and isinstance(existing, nn.Linear):
                            in_feat = existing.in_features
                            out_feat = int(ck_shape[0])
                            if out_feat != existing.out_features:
                                if verbose:
                                    print(f"Replacing {mod_path}: Linear({in_feat}->{existing.out_features}) -> Linear({in_feat}->{out_feat})")
                                try:
                                    new_mod = nn.Linear(in_feat, out_feat).to(device)
                                except Exception:
                                    new_mod = nn.Linear(in_feat, out_feat)
                                try:
                                    with torch.no_grad():
                                        nc = min(existing.out_features, out_feat)
                                        dev = next(new_mod.parameters()).device
                                        new_mod.weight[:nc, :].copy_(existing.weight[:nc, :].to(dev))
                                        if existing.bias is not None:
                                            new_mod.bias[:nc].copy_(existing.bias[:nc].to(dev))
                                except Exception:
                                    pass
                                try:
                                    _set_module_by_path(target, mod_path, new_mod)
                                    replaced.append(mod_path)
                                except Exception as e:
                                    if verbose:
                                        print(f"Failed to replace {mod_path}: {e}")
                if replaced and verbose:
                    print("Replaced modules:", replaced)
                    target_state = target.state_dict()
                    filtered = {k: v for k, v in state.items() if k in target_state and getattr(v, "shape", None) == getattr(target_state[k], "shape", None)}
            except Exception:
                pass
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
        if verbose:
            print(f"load_state_dict failed: {e}")
        try:
            fixed = {(k.replace("module.", "")): v for k, v in state.items()}
            try:
                target_state = target.state_dict()
                filtered = {k: v for k, v in fixed.items() if k in target_state and getattr(v, "shape", None) == getattr(target_state[k], "shape", None)}
            except Exception:
                filtered = fixed
            target.load_state_dict(filtered, strict=False)
            try:
                if hasattr(target, "to") and callable(getattr(target, "to")):
                    target.to(device)
            except Exception:
                pass
            return True, None
        except Exception as e2:
            if verbose:
                print(f"Second load attempt failed: {e2}")
            return False, None


def _tensor_to_numpy(t):
    try:
        return t.detach().cpu().numpy()
    except Exception:
        return np.asarray(t)


def _normalize_masks_to_bool(masks: Any, score_thresh: float = 0.0) -> Optional[List[np.ndarray]]:
    """Accept masks in many shapes/types and return list of boolean HxW arrays."""
    if masks is None:
        return None

    # supervision sometimes uses "mask" with shape (N,H,W) or list of arrays
    if isinstance(masks, list):
        out = []
        for m in masks:
            mm = _tensor_to_numpy(m)
            if mm.ndim == 3 and mm.shape[0] == 1:
                mm = mm[0]
            if mm.ndim != 2:
                continue
            out.append(mm > score_thresh)
        return out if out else None

    mm = _tensor_to_numpy(masks)

    # (H,W)
    if mm.ndim == 2:
        return [mm > score_thresh]

    # (N,H,W)
    if mm.ndim == 3:
        return [(mm[i] > score_thresh) for i in range(mm.shape[0])]

    # (N,1,H,W)
    if mm.ndim == 4 and mm.shape[1] == 1:
        return [(mm[i, 0] > score_thresh) for i in range(mm.shape[0])]

    return None


def parse_model_output(
    output,
    img_w: int,
    img_h: int,
    score_thresh: float = 0.3,
    return_masks: bool = False,
    mask_thresh: float = 0.5,
):
    """Parse common model output formats.

    Returns:
      - if return_masks=False: (boxes, scores, labels)
      - if return_masks=True:  (boxes, scores, labels, masks)

    boxes are [x1,y1,x2,y2] in absolute pixels.
    masks (if present) are a list of boolean numpy arrays HxW (frame-sized if already matched).
    """
    boxes: List[List[float]] = []
    scores: List[float] = []
    labels: List[int] = []
    masks: Optional[List[np.ndarray]] = None

    # supervision-like objects
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
                    cx = tmp[:, 0]
                    cy = tmp[:, 1]
                    w_ = tmp[:, 2]
                    h_ = tmp[:, 3]
                    x1 = cx - w_ / 2
                    y1 = cy - h_ / 2
                    x2 = cx + w_ / 2
                    y2 = cy + h_ / 2
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

            if return_masks:
                # common names for masks on supervision objects
                m = None
                for cand in ("mask", "masks", "segmentation", "segmentations"):
                    if hasattr(output, cand):
                        m = getattr(output, cand)
                        break
                masks = _normalize_masks_to_bool(m, score_thresh=float(mask_thresh))

            if b is not None and b.size:
                for i in range(b.shape[0]):
                    bx = b[i]
                    if float(bx.max()) <= 1.0:
                        bx_pixels = [float(bx[0] * img_w), float(bx[1] * img_h), float(bx[2] * img_w), float(bx[3] * img_h)]
                    else:
                        bx_pixels = [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])]
                    if bx_pixels[2] < bx_pixels[0] or bx_pixels[3] < bx_pixels[1]:
                        cx = bx_pixels[0]
                        cy = bx_pixels[1]
                        w_ = bx_pixels[2]
                        h_ = bx_pixels[3]
                        x1 = cx - w_ / 2
                        y1 = cy - h_ / 2
                        x2 = cx + w_ / 2
                        y2 = cy + h_ / 2
                        bx_pixels = [x1, y1, x2, y2]

                    sc = float(s[i]) if s is not None and i < len(s) else 1.0
                    if sc < score_thresh:
                        continue

                    boxes.append(bx_pixels)
                    scores.append(sc)
                    lab = int(l[i]) if l is not None and i < len(l) else 0
                    labels.append(int(lab))

                if return_masks:
                    # If masks exist but were not filtered, align lengths best-effort
                    if masks is not None and len(masks) != len(boxes):
                        # naive alignment: truncate to min
                        n = min(len(masks), len(boxes))
                        masks = masks[:n]
                        boxes[:] = boxes[:n]
                        scores[:] = scores[:n]
                        labels[:] = labels[:n]
                    return boxes, scores, labels, masks
                return boxes, scores, labels
        except Exception:
            pass

    # dict / tensor outputs
    try:
        out = output[0] if isinstance(output, (list, tuple)) else output

        if isinstance(out, dict):
            # masks possibly here
            if return_masks:
                m = None
                for k in ("masks", "mask", "pred_masks", "segmentation"):
                    if k in out:
                        m = out[k]
                        break
                masks = _normalize_masks_to_bool(m, score_thresh=float(mask_thresh))

            if "boxes" in out:
                b = out["boxes"]
                s = out.get("scores", None)
                l = out.get("labels", out.get("classes", None))
                if isinstance(b, torch.Tensor):
                    b = b.detach().cpu()
                    if b.numel() == 0:
                        if return_masks:
                            return [], [], [], None
                        return [], [], []
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

                    if return_masks and masks is not None:
                        masks = [masks[i] for i in keep if i < len(masks)]

                    if return_masks:
                        return boxes, scores, labels, masks
                    return boxes, scores, labels

            if "pred_boxes" in out:
                pb = out["pred_boxes"]
                if isinstance(pb, torch.Tensor):
                    pb_cpu = pb.detach().cpu()
                    maxv = float(pb_cpu.max())
                    if maxv <= 1.0:
                        cx = pb_cpu[:, 0]
                        cy = pb_cpu[:, 1]
                        bw = pb_cpu[:, 2]
                        bh = pb_cpu[:, 3]
                        x1 = (cx - bw / 2.0) * img_w
                        y1 = (cy - bh / 2.0) * img_h
                        x2 = (cx + bw / 2.0) * img_w
                        y2 = (cy + bh / 2.0) * img_h
                        xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                    else:
                        xyxy = pb_cpu

                    s = out.get("scores", out.get("pred_scores", None))
                    l = out.get("labels", out.get("pred_labels", None))
                    if s is None:
                        s = torch.ones((xyxy.shape[0],))
                    else:
                        s = s.detach().cpu()
                    if l is None:
                        l = torch.zeros((xyxy.shape[0],), dtype=torch.int64)
                    else:
                        l = l.detach().cpu().long()

                    keep = []
                    for i in range(xyxy.shape[0]):
                        if float(s[i].item()) < score_thresh:
                            continue
                        keep.append(i)
                        boxes.append([float(x) for x in xyxy[i].tolist()])
                        scores.append(float(s[i].item()))
                        labels.append(int(l[i].item()))

                    if return_masks and masks is not None:
                        masks = [masks[i] for i in keep if i < len(masks)]

                    if return_masks:
                        return boxes, scores, labels, masks
                    return boxes, scores, labels

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
                if return_masks:
                    return boxes, scores, labels, None
                return boxes, scores, labels
    except Exception:
        pass

    if return_masks:
        return [], [], [], None
    return [], [], []


def detections_to_json(
    boxes: List[List[float]],
    scores: List[float],
    labels: List[int],
    class_names: Optional[List[str]] = None,
    image_id: Optional[str] = None,
    score_thresh: float = 0.3,
):
    out = {"image_id": image_id, "detections": []}
    for i in range(len(boxes)):
        sc = float(scores[i])
        if sc < score_thresh:
            continue
        b = [float(x) for x in boxes[i]]
        lid = int(labels[i])
        lname = (class_names[lid] if class_names and 0 <= lid < len(class_names) else str(lid))
        out["detections"].append({"bbox": b, "score": sc, "label_id": lid, "label_name": lname})
    return out
