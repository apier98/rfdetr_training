from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import os


@dataclass(frozen=True)
class LoadResult:
    ok: bool
    replacement_model: Optional[object] = None
    message: str = ""


def _torch_load(path: str, map_location: Any, *, verbose: bool) -> object:
    import torch

    try:
        return torch.load(path, map_location=map_location)
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

        # PyTorch 2.6 defaults weights_only=True; allowlist argparse.Namespace first.
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
                        ckpt = torch.load(path, map_location=map_location, weights_only=True)
            except Exception as e_safe:
                if verbose:
                    print("Safe weights_only load failed:", e_safe)

        if ckpt is None:
            if verbose:
                print("Retrying torch.load with weights_only=False (this may execute code from the checkpoint).")
            if supports_weights_only:
                return torch.load(path, map_location=map_location, weights_only=False)
            return torch.load(path, map_location=map_location)

        return ckpt


def _find_state_dict(ckpt: object, checkpoint_key: Optional[str]) -> Optional[dict]:
    if not isinstance(ckpt, dict):
        return None

    if checkpoint_key and checkpoint_key in ckpt and isinstance(ckpt[checkpoint_key], dict):
        return ckpt[checkpoint_key]

    for key in ("model_state_dict", "state_dict", "model", "net"):
        val = ckpt.get(key)
        if isinstance(val, dict):
            return val

    # sometimes the checkpoint dict itself is a state_dict
    try:
        if any(
            str(k).startswith("backbone")
            or str(k).startswith("transformer")
            or "class_embed" in str(k)
            for k in ckpt.keys()
        ):
            return ckpt  # type: ignore[return-value]
    except Exception:
        pass

    return None


def load_checkpoint_weights(
    model: object,
    path: str,
    device: object,
    *,
    checkpoint_key: Optional[str] = None,
    allow_replace_model: bool = False,
    verbose: bool = False,
) -> LoadResult:
    if not path:
        return LoadResult(False, None, "Empty checkpoint path")
    if not os.path.exists(path):
        return LoadResult(False, None, f"Weights file not found: {path}")

    ckpt = _torch_load(path, map_location=device, verbose=verbose)

    # If explicitly allowed, prefer returning a pickled model object (exact architecture).
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
                    return LoadResult(True, maybe, f"replacement:{candidate}")
            except Exception:
                pass
            if hasattr(maybe, "state_dict") and callable(getattr(maybe, "state_dict")):
                if verbose:
                    print(f"Using checkpoint['{candidate}'] as replacement model ({type(maybe)}).")
                return LoadResult(True, maybe, f"replacement:{candidate}")

    state = _find_state_dict(ckpt, checkpoint_key)
    if state is None:
        # last resort: let wrapper handle it
        if hasattr(model, "load") and callable(getattr(model, "load")):
            try:
                model.load(path)  # type: ignore[attr-defined]
                return LoadResult(True, None, "loaded_via_wrapper_load()")
            except Exception as e:
                return LoadResult(False, None, f"Could not load checkpoint: {e}")
        return LoadResult(False, None, "Could not autodetect state_dict in checkpoint")

    # choose target for load_state_dict
    target = None
    if hasattr(model, "load_state_dict") and callable(getattr(model, "load_state_dict")):
        target = model
    else:
        for attr in ("model", "net", "network", "module", "detector", "backbone", "transformer"):
            inner = getattr(model, attr, None)
            if inner is not None and hasattr(inner, "load_state_dict") and callable(getattr(inner, "load_state_dict")):
                target = inner
                break
    if target is None:
        return LoadResult(False, None, "Model does not support load_state_dict")

    # filter by matching shapes (avoid hard crashes on mismatched heads/architectures)
    try:
        target_state = target.state_dict()  # type: ignore[attr-defined]
        filtered = {}
        skipped = 0
        for k, v in state.items():
            if k not in target_state:
                skipped += 1
                continue
            try:
                if getattr(v, "shape", None) == getattr(target_state[k], "shape", None):
                    filtered[k] = v
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        if verbose:
            print(f"Loading {len(filtered)} matched tensors from checkpoint; skipping {skipped} mismatched tensors")
        target.load_state_dict(filtered, strict=False)  # type: ignore[attr-defined]
        try:
            if hasattr(target, "to") and callable(getattr(target, "to")):
                target.to(device)  # type: ignore[attr-defined]
        except Exception:
            pass
        return LoadResult(True, None, f"loaded_filtered_tensors:{len(filtered)}")
    except Exception as e:
        return LoadResult(False, None, f"load_state_dict failed: {e}")

