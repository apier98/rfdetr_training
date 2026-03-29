from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple

import os


@dataclass(frozen=True)
class LoadResult:
    ok: bool
    replacement_model: Optional[object] = None
    message: str = ""
    missing_keys: Tuple[str, ...] = ()
    unexpected_keys: Tuple[str, ...] = ()
    mismatched_shapes: Tuple[Tuple[str, str, str], ...] = ()  # (key, expected, got)


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


def extract_state_dict_from_checkpoint(
    path: str,
    *,
    device: object,
    checkpoint_key: Optional[str] = None,
    verbose: bool = False,
) -> Optional[Dict[str, object]]:
    """Extract a plain (tensor-only) state_dict from an arbitrary checkpoint.

    This is intended for creating deployment-friendly "weights-only" checkpoints that
    load cleanly under PyTorch 2.6+ defaults (weights_only=True).
    """
    if not path:
        return None
    if not os.path.exists(path):
        return None

    ckpt = _torch_load(path, map_location=device, verbose=verbose)
    state = _find_state_dict(ckpt, checkpoint_key)
    if not isinstance(state, dict):
        return None

    try:
        import torch
    except Exception:
        return None

    out: Dict[str, object] = {}
    skipped = 0
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[str(k)] = v
        else:
            skipped += 1

    if verbose and skipped:
        print(f"Note: extracted state_dict tensors={len(out)}; skipped_non_tensors={skipped}")
    return out


def save_portable_checkpoint(
    *,
    src_path: str,
    dst_path: str,
    device: object,
    checkpoint_key: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """Save a weights-only checkpoint (tensor-only state_dict) at dst_path."""
    try:
        import torch
    except Exception as e:
        return False, f"torch not importable: {e}"

    state = extract_state_dict_from_checkpoint(
        src_path,
        device=device,
        checkpoint_key=checkpoint_key,
        verbose=verbose,
    )
    if not state:
        return False, "Could not extract a tensor-only state_dict from checkpoint"

    try:
        payload: Dict[str, object] = {
            "format_version": 1,
            "state_dict": state,
        }
        torch.save(payload, dst_path)
        return True, f"Wrote portable checkpoint: {dst_path}"
    except Exception as e:
        return False, f"torch.save failed: {e}"


def load_checkpoint_weights(
    model: object,
    path: str,
    device: object,
    *,
    checkpoint_key: Optional[str] = None,
    allow_replace_model: bool = False,
    strict: bool = False,
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
                    if strict:
                        return LoadResult(
                            False,
                            None,
                            "strict=True disallows replacement model loading (pickled model objects are not deployment-friendly). "
                            "Re-run without --strict if you trust this checkpoint.",
                        )
                    return LoadResult(True, maybe, f"replacement:{candidate}")
            except Exception:
                pass
            if hasattr(maybe, "state_dict") and callable(getattr(maybe, "state_dict")):
                if verbose:
                    print(f"Using checkpoint['{candidate}'] as replacement model ({type(maybe)}).")
                if strict:
                    return LoadResult(
                        False,
                        None,
                        "strict=True disallows replacement model loading (pickled model objects are not deployment-friendly). "
                        "Re-run without --strict if you trust this checkpoint.",
                    )
                return LoadResult(True, maybe, f"replacement:{candidate}")

    state = _find_state_dict(ckpt, checkpoint_key)
    if state is None:
        # last resort: let wrapper handle it
        if hasattr(model, "load") and callable(getattr(model, "load")):
            try:
                if strict:
                    return LoadResult(False, None, "strict=True requires a state_dict; wrapper .load(path) is not allowed")
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
        # Common wrappers (including `rfdetr`): the actual nn.Module is often nested.
        try:
            from .torch_compat import unwrap_torch_module

            target = unwrap_torch_module(model)
        except Exception:
            target = None

    if target is None or not hasattr(target, "load_state_dict") or not callable(getattr(target, "load_state_dict")):
        return LoadResult(False, None, "Model does not support load_state_dict")

    # Prepare a compatibility report.
    try:
        target_state = target.state_dict()  # type: ignore[attr-defined]

        missing: List[str] = []
        unexpected: List[str] = []
        mismatched: List[Tuple[str, str, str]] = []

        # missing keys (expected by model, absent in checkpoint)
        for k in target_state.keys():
            if k not in state:
                missing.append(str(k))

        # unexpected keys and mismatched shapes
        for k, v in state.items():
            if k not in target_state:
                unexpected.append(str(k))
                continue
            try:
                exp = getattr(target_state[k], "shape", None)
                got = getattr(v, "shape", None)
                if exp is not None and got is not None and tuple(exp) != tuple(got):
                    mismatched.append((str(k), str(tuple(exp)), str(tuple(got))))
            except Exception:
                # if shape inspection fails, treat as mismatch in strict mode
                mismatched.append((str(k), "unknown", "unknown"))

        if strict:
            if missing or unexpected or mismatched:
                # Try a targeted fix for common RF-DETR custom-dataset checkpoints:
                # resize class head layers to match state_dict shapes.
                if not missing and not unexpected and mismatched:
                    try:
                        from .torch_compat import maybe_resize_rfdetr_class_heads_for_state_dict

                        changed = maybe_resize_rfdetr_class_heads_for_state_dict(target, state, verbose=verbose)
                        if changed:
                            # recompute mismatch report after resizing
                            target_state = target.state_dict()  # type: ignore[attr-defined]
                            missing = [str(k) for k in target_state.keys() if k not in state]
                            unexpected = [str(k) for k in state.keys() if k not in target_state]
                            mismatched = []
                            for k, v in state.items():
                                if k not in target_state:
                                    continue
                                try:
                                    exp = getattr(target_state[k], "shape", None)
                                    got = getattr(v, "shape", None)
                                    if exp is not None and got is not None and tuple(exp) != tuple(got):
                                        mismatched.append((str(k), str(tuple(exp)), str(tuple(got))))
                                except Exception:
                                    mismatched.append((str(k), "unknown", "unknown"))
                    except Exception:
                        pass

                if missing or unexpected or mismatched:
                    # Fail fast for deployment: partial loads are a foot-gun.
                    parts: List[str] = ["Strict checkpoint load failed:"]
                    if missing:
                        parts.append(f"- missing keys: {len(missing)}")
                    if unexpected:
                        parts.append(f"- unexpected keys: {len(unexpected)}")
                    if mismatched:
                        parts.append(f"- mismatched shapes: {len(mismatched)}")
                    # show a small sample so the user can diagnose quickly
                    if missing:
                        parts.append(f"- missing sample: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
                    if unexpected:
                        parts.append(
                            f"- unexpected sample: {', '.join(unexpected[:10])}{' ...' if len(unexpected) > 10 else ''}"
                        )
                    if mismatched:
                        sm = mismatched[:10]
                        parts.append(
                            "- mismatched sample: "
                            + ", ".join([f"{k} exp={e} got={g}" for (k, e, g) in sm])
                            + (" ..." if len(mismatched) > 10 else "")
                        )
                    return LoadResult(
                        False,
                        None,
                        "\n".join(parts),
                        missing_keys=tuple(missing),
                        unexpected_keys=tuple(unexpected),
                        mismatched_shapes=tuple(mismatched),
                    )

            if verbose:
                print("Strict load: checkpoint and model state_dicts match. Loading with strict=True.")
            target.load_state_dict(state, strict=True)  # type: ignore[attr-defined]
        else:
            # filter by matching shapes (avoid hard crashes on mismatched heads/architectures)
            filtered: Dict[str, object] = {}
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
                if mismatched:
                    sm = mismatched[:10]
                    print(
                        "Mismatched shapes sample:",
                        ", ".join([f"{k} exp={e} got={g}" for (k, e, g) in sm]) + (" ..." if len(mismatched) > 10 else ""),
                    )
            target.load_state_dict(filtered, strict=False)  # type: ignore[attr-defined]
        try:
            if hasattr(target, "to") and callable(getattr(target, "to")):
                target.to(device)  # type: ignore[attr-defined]
        except Exception:
            pass
        msg = "loaded_strict_state_dict" if strict else f"loaded_filtered_tensors:{len(filtered)}"
        return LoadResult(
            True,
            None,
            msg,
            missing_keys=tuple(missing),
            unexpected_keys=tuple(unexpected),
            mismatched_shapes=tuple(mismatched),
        )
    except Exception as e:
        return LoadResult(False, None, f"load_state_dict failed: {e}")
