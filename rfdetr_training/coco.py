from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import json


@dataclass(frozen=True)
class CocoValidation:
    ok: bool
    errors: List[str]
    warnings: List[str]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _segmentation_is_nonempty(seg: object) -> bool:
    # COCO polygons: list[list[float]] with len>=1 and each polygon has >=6 numbers
    if isinstance(seg, list):
        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6:
                return True
        return False
    # RLE dict, treat as present if non-empty dict
    if isinstance(seg, dict):
        return bool(seg)
    return False


def validate_coco_split(
    split_dir: Path,
    *,
    task: str,
    check_images_exist: bool = True,
) -> CocoValidation:
    errors: List[str] = []
    warnings: List[str] = []

    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return CocoValidation(False, [f"Missing: {ann_path}"], [])

    try:
        coco = _load_json(ann_path)
    except Exception as e:
        return CocoValidation(False, [f"Failed to parse {ann_path}: {e}"], [])

    images = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []
    cats = coco.get("categories", []) or []

    if not isinstance(images, list) or not isinstance(anns, list) or not isinstance(cats, list):
        errors.append(f"Invalid COCO structure in {ann_path} (images/annotations/categories must be lists)")
        return CocoValidation(False, errors, warnings)

    if len(images) == 0:
        warnings.append(f"{ann_path} has 0 images")
    if len(anns) == 0:
        warnings.append(f"{ann_path} has 0 annotations")
    if len(cats) == 0:
        warnings.append(f"{ann_path} has 0 categories")

    image_ids: Set[int] = set()
    file_names: Set[str] = set()
    for im in images:
        try:
            iid = int(im.get("id"))
            image_ids.add(iid)
        except Exception:
            errors.append(f"{ann_path}: image missing int id: {im}")
            continue
        fn = str(im.get("file_name", "")).strip()
        if not fn:
            errors.append(f"{ann_path}: image id={iid} missing file_name")
        else:
            file_names.add(fn)
            if check_images_exist and not (split_dir / fn).exists():
                warnings.append(f"{ann_path}: missing image file on disk: {split_dir / fn}")

    cat_ids: Set[int] = set()
    for c in cats:
        try:
            cat_ids.add(int(c.get("id")))
        except Exception:
            errors.append(f"{ann_path}: category missing int id: {c}")

    seg_nonempty_count = 0
    for a in anns:
        try:
            iid = int(a.get("image_id"))
        except Exception:
            errors.append(f"{ann_path}: annotation missing int image_id: {a}")
            continue
        if iid not in image_ids:
            errors.append(f"{ann_path}: annotation references unknown image_id={iid}")

        try:
            cid = int(a.get("category_id"))
        except Exception:
            errors.append(f"{ann_path}: annotation missing int category_id: {a}")
            continue
        if cat_ids and cid not in cat_ids:
            warnings.append(f"{ann_path}: annotation category_id={cid} not present in categories[]")

        if task == "seg":
            seg = a.get("segmentation", None)
            if _segmentation_is_nonempty(seg):
                seg_nonempty_count += 1

    if task == "seg" and len(anns) > 0 and seg_nonempty_count == 0:
        warnings.append(f"{ann_path}: task=seg but 0 annotations have non-empty segmentation")

    return CocoValidation(len(errors) == 0, errors, warnings)


def ensure_minimal_test_split(coco_dir: Path) -> Optional[Path]:
    test_dir = coco_dir / "test"
    ann_path = test_dir / "_annotations.coco.json"
    if ann_path.exists():
        return None

    # Try to reuse categories from valid/train if present.
    categories: list = []
    for candidate in (coco_dir / "valid" / "_annotations.coco.json", coco_dir / "train" / "_annotations.coco.json"):
        if not candidate.exists():
            continue
        try:
            categories = _load_json(candidate).get("categories", []) or []
            break
        except Exception:
            continue

    test_dir.mkdir(parents=True, exist_ok=True)
    minimal = {
        "info": {"description": "auto-generated empty test split", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    ann_path.write_text(json.dumps(minimal, indent=2), encoding="utf-8")
    return ann_path

