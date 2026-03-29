from __future__ import annotations

from typing import Optional


def unwrap_torch_module(obj: object):
    """Best-effort: find the underlying torch.nn.Module inside wrapper objects.

    `rfdetr` models are wrappers; the actual nn.Module lives at `.model.model`.
    """

    try:
        import torch.nn as nn
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PyTorch not available: {e}")

    if isinstance(obj, nn.Module):
        return obj

    # Common `rfdetr` wrapper nesting: outer.model (rfdetr.main.Model) -> .model (nn.Module)
    inner = getattr(obj, "model", None)
    if inner is not None:
        if isinstance(inner, nn.Module):
            return inner
        inner2 = getattr(inner, "model", None)
        if isinstance(inner2, nn.Module):
            return inner2

    # Generic attribute scan (shallow, but covers most wrappers)
    for attr in ("net", "network", "module", "detector", "backbone", "transformer", "inner"):
        val = getattr(obj, attr, None)
        if isinstance(val, nn.Module):
            return val
        if val is not None:
            maybe = getattr(val, "model", None)
            if isinstance(maybe, nn.Module):
                return maybe

    # best-effort: search one level deep through __dict__
    try:
        for _, val in vars(obj).items():
            if isinstance(val, nn.Module):
                return val
            maybe = getattr(val, "model", None)
            if isinstance(maybe, nn.Module):
                return maybe
    except Exception:
        pass

    raise TypeError(f"Could not find a torch.nn.Module inside model object of type {type(obj)}")


def infer_backbone_patch_size(module) -> Optional[int]:
    """Try to infer the backbone patch size (divisibility constraint) from a model."""

    try:
        # Search submodules for an integer `patch_size` attribute.
        for sub in module.modules():
            ps = getattr(sub, "patch_size", None)
            if isinstance(ps, int) and 1 <= ps <= 1024:
                return int(ps)
    except Exception:
        return None
    return None


def maybe_resize_rfdetr_class_heads_for_state_dict(module, state_dict: dict, *, verbose: bool = False) -> bool:
    """Resize known RF-DETR classification heads to match a checkpoint state_dict.

    This is needed because the installed `rfdetr` wrappers often instantiate with COCO head shapes
    (e.g. 91), while workspace training checkpoints can have `class_embed.*` shaped to a custom
    dataset (e.g. 1).

    Returns True if any module was resized.
    """

    import torch
    import torch.nn as nn

    w = state_dict.get("class_embed.weight")
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return False

    out_features = int(w.shape[0])
    in_features = int(w.shape[1])

    changed = False

    ce = getattr(module, "class_embed", None)
    if isinstance(ce, nn.Linear):
        if int(ce.in_features) == in_features and int(ce.out_features) != out_features:
            if verbose:
                print(f"Resizing class_embed: {ce.out_features} -> {out_features}")
            module.class_embed = nn.Linear(in_features, out_features, bias=True)
            changed = True

    tr = getattr(module, "transformer", None)
    enc_ce = getattr(tr, "enc_out_class_embed", None)
    if isinstance(enc_ce, nn.ModuleList):
        for i in range(len(enc_ce)):
            li = enc_ce[i]
            if isinstance(li, nn.Linear) and int(li.in_features) == in_features and int(li.out_features) != out_features:
                if verbose:
                    print(f"Resizing transformer.enc_out_class_embed[{i}]: {li.out_features} -> {out_features}")
                enc_ce[i] = nn.Linear(in_features, out_features, bias=True)
                changed = True

    return changed
