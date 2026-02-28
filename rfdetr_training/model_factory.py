from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _ctor_best_effort(
    cls: type,
    *,
    num_classes: Optional[int],
    pretrain_weights: Optional[str],
    force_pretrain_weights_kwarg: bool,
) -> Any:
    kwargs: Dict[str, Any] = {}
    # Important: many `rfdetr` classes default to downloading pretrained weights
    # when `pretrain_weights` is omitted. Passing `None` explicitly is our best
    # effort way to disable downloads for deployment/inference bundles.
    if force_pretrain_weights_kwarg:
        kwargs["pretrain_weights"] = pretrain_weights
    elif pretrain_weights is not None:
        kwargs["pretrain_weights"] = pretrain_weights
    if num_classes is not None:
        kwargs["num_classes"] = int(num_classes)
    try:
        return cls(**kwargs) if kwargs else cls()
    except TypeError:
        # Some versions of rfdetr constructors may not accept these kwargs.
        try:
            kwargs.pop("pretrain_weights", None)
            return cls(**kwargs) if kwargs else cls()
        except TypeError:
            return cls()


def _size_to_class_suffix(size: str) -> str:
    s = (size or "").strip().lower()
    if s in ("nano", "small", "base", "medium", "large"):
        return s.capitalize()
    if s == "xlarge":
        return "XLarge"
    if s in ("2xlarge", "2xl", "2x"):
        return "2XLarge"
    # best-effort fallback
    return s.capitalize()


def instantiate_rfdetr_model(
    task: str,
    size: str,
    num_classes: Optional[int] = None,
    pretrain_weights: Optional[str] = None,
    *,
    force_pretrain_weights_kwarg: bool = True,
) -> Tuple[Any, str, bool]:
    """
    Instantiate an RF-DETR model from the installed `rfdetr` package.

    Returns: (model, class_name, size_applied)
      - size_applied is False when `size` had no effect (e.g. segmentation preview fallback).
    """
    task_norm = (task or "detect").lower().strip()
    size_norm = (size or "nano").lower().strip()
    suffix = _size_to_class_suffix(size_norm)

    # Import lazily so CLI tooling can function even when rfdetr isn't installed.
    import rfdetr  # type: ignore

    if task_norm == "seg":
        # Prefer size-specific segmentation classes when available; fall back to preview.
        candidates = [
            f"RFDETRSeg{suffix}",
            f"RFDETR{suffix}Seg",
        ]
        for name in candidates:
            cls = getattr(rfdetr, name, None)
            if isinstance(cls, type):
                model = _ctor_best_effort(
                    cls,
                    num_classes=num_classes,
                    pretrain_weights=pretrain_weights,
                    force_pretrain_weights_kwarg=force_pretrain_weights_kwarg,
                )
                return model, name, True

        preview = getattr(rfdetr, "RFDETRSegPreview", None)
        if isinstance(preview, type):
            model = _ctor_best_effort(
                preview,
                num_classes=num_classes,
                pretrain_weights=pretrain_weights,
                force_pretrain_weights_kwarg=force_pretrain_weights_kwarg,
            )
            return model, "RFDETRSegPreview", False

        raise ValueError(
            "Segmentation model class not found in installed `rfdetr`. "
            "Expected one of: RFDETRSeg<Nano|Small|Base|Medium> or RFDETRSegPreview."
        )

    # detect (RF-DETR docs include Nano..2XLarge; some may require rfdetr[plus])
    name = f"RFDETR{suffix}"
    cls = getattr(rfdetr, name, None)
    if not isinstance(cls, type):
        hint = ""
        if size_norm in ("xlarge", "2xlarge", "2xl", "2x"):
            hint = " If you need XLarge/2XLarge, install the extra (e.g. `pip install rfdetr[plus]`)."
        raise ValueError(f"Model class {name} not found in installed `rfdetr`.{hint}")
    model = _ctor_best_effort(
        cls,
        num_classes=num_classes,
        pretrain_weights=pretrain_weights,
        force_pretrain_weights_kwarg=force_pretrain_weights_kwarg,
    )
    return model, name, True
