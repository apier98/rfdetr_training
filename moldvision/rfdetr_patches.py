from __future__ import annotations

from typing import Any


def patch_albumentations_empty_masks() -> bool:
    """Patch RF-DETR to tolerate empty seg masks during Albumentations transforms.

    Some RF-DETR + Albumentations combinations raise ``ValueError: masks cannot be empty``
    when a segmentation sample has zero instances. That makes background-only images
    unusable for segmentation training even though they are valid negative examples.
    """
    try:
        import torch
        from rfdetr.datasets.transforms import AlbumentationsWrapper
    except Exception:
        return False

    if getattr(AlbumentationsWrapper, "_moldvision_empty_masks_patched", False):
        return False

    original = AlbumentationsWrapper._apply_geometric_transform

    def _is_empty_masks(masks: Any) -> bool:
        if torch.is_tensor(masks):
            return masks.ndim >= 1 and int(masks.shape[0]) == 0
        try:
            return len(masks) == 0
        except Exception:
            return False

    def _patched_apply_geometric_transform(self: Any, image_np: Any, target: Any, labels: Any) -> Any:
        masks = target.get("masks") if isinstance(target, dict) else None
        if masks is None or not _is_empty_masks(masks):
            return original(self, image_np, target, labels)

        target_no_masks = dict(target)
        target_no_masks.pop("masks", None)
        image_out, target_out = original(self, image_np, target_no_masks, labels)
        width, height = image_out.size
        target_out["masks"] = torch.zeros((0, height, width), dtype=torch.bool)
        return image_out, target_out

    AlbumentationsWrapper._apply_geometric_transform = _patched_apply_geometric_transform
    AlbumentationsWrapper._moldvision_empty_masks_patched = True
    return True
