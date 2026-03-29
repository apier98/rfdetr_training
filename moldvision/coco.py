from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from datetime import datetime
import shutil
from .pathutil import resolve_path

from .jsonutil import load_json_strict, save_json


@dataclass(frozen=True)
class CocoValidation:
    ok: bool
    errors: List[str]
    warnings: List[str]


@dataclass(frozen=True)
class CocoPruneResult:
    ok: bool
    removed_images: int = 0
    removed_annotations: int = 0
    backup_path: Optional[Path] = None
    message: str = ""


def _segmentation_is_nonempty(seg: object) -> bool:
    # COCO polygons: list[list[float]] with len>=1 and each polygon has >=6 numbers
    if isinstance(seg, list):
        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6:
                return True
        return False
    # RLE dict, treat as present if non-empty dict
    if isinstance(seg, dict):
        counts = seg.get("counts")
        size = seg.get("size")
        if counts is None or size is None:
            return False
        if isinstance(size, (list, tuple)) and len(size) == 2:
            try:
                h = int(size[0])
                w = int(size[1])
                if h <= 0 or w <= 0:
                    return False
            except Exception:
                return False
        else:
            return False
        if isinstance(counts, str):
            return len(counts) > 0
        if isinstance(counts, (list, tuple, bytes, bytearray)):
            return len(counts) > 0
        return False
    return False


def _segmentation_is_valid(seg: object) -> bool:
    # polygons
    if isinstance(seg, list):
        if len(seg) == 0:
            return False
        for poly in seg:
            if not isinstance(poly, list):
                return False
            if len(poly) < 6 or (len(poly) % 2) != 0:
                return False
        return True
    # RLE dict
    if isinstance(seg, dict):
        counts = seg.get("counts")
        size = seg.get("size")
        if counts is None or size is None:
            return False
        if isinstance(size, (list, tuple)) and len(size) == 2:
            try:
                h = int(size[0])
                w = int(size[1])
            except Exception:
                return False
            if h <= 0 or w <= 0:
                return False
        else:
            return False
        # pycocotools allows counts to be a compact string or a list.
        if isinstance(counts, str):
            return len(counts) > 0
        if isinstance(counts, (list, tuple, bytes, bytearray)):
            return len(counts) > 0
        return False
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
        coco = load_json_strict(ann_path)
    except Exception as e:
        return CocoValidation(False, [f"Failed to parse {ann_path}: {e}"], [])

    images = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []
    cats = coco.get("categories", []) or []

    if not isinstance(images, list) or not isinstance(anns, list) or not isinstance(cats, list):
        errors.append(f"Invalid COCO structure in {ann_path} (images/annotations/categories must be lists)")
        return CocoValidation(False, errors, warnings)

    # category id sanity (common RF-DETR issue: 1-indexed COCO or holes)
    cat_ids: Set[int] = set()
    cat_names: List[str] = []
    for c in cats:
        try:
            cat_ids.add(int(c.get("id")))
            cat_names.append(str(c.get("name", "")).strip())
        except Exception:
            errors.append(f"{ann_path}: category missing int id: {c}")

    if cat_ids:
        min_id = min(cat_ids)
        max_id = max(cat_ids)
        # RF-DETR expects COCO-style ids, but training has had bugs with 1-indexed datasets.
        if min_id == 1:
            warnings.append(
                f"{ann_path}: categories are 1-indexed (min id=1). "
                "RF-DETR has had training issues with 1-indexed COCO. Prefer normalizing to 0..N-1."
            )
        elif min_id != 0:
            warnings.append(f"{ann_path}: categories min id is {min_id} (expected 0 for smoothest training).")

        expected = set(range(min_id, max_id + 1))
        holes = sorted(expected.difference(cat_ids))
        if holes:
            warnings.append(
                f"{ann_path}: category ids have holes (missing ids like {holes[:10]}{'...' if len(holes)>10 else ''}). "
                "This can cause num_classes mismatches. Prefer normalizing to 0..N-1."
            )

        # duplicate category names can lead to wrong num_classes/class_names in some RF-DETR versions
        name_counts = {}
        for n in cat_names:
            if not n:
                continue
            name_counts[n] = name_counts.get(n, 0) + 1
        dups = [n for n, ct in name_counts.items() if ct > 1]
        if dups:
            warnings.append(
                f"{ann_path}: duplicate category names found ({dups[:10]}{'...' if len(dups)>10 else ''}). "
                "This can cause duplicated class_names / wrong num_classes. Prefer aligning categories to METADATA.json."
            )

    if len(images) == 0:
        warnings.append(f"{ann_path} has 0 images")
    if len(anns) == 0:
        warnings.append(f"{ann_path} has 0 annotations")
    if len(cats) == 0:
        warnings.append(f"{ann_path} has 0 categories")
    else:
        # Some RF-DETR versions assume every category has a 'supercategory' field.
        missing_sc = 0
        for c in cats:
            if isinstance(c, dict) and "supercategory" not in c:
                missing_sc += 1
        if missing_sc:
            warnings.append(f"{ann_path}: {missing_sc} categories are missing 'supercategory' (will crash some RF-DETR versions)")

    image_ids: Set[int] = set()
    image_id_to_name: Dict[int, str] = {}
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
            image_id_to_name[iid] = fn
            if check_images_exist and not (split_dir / fn).exists():
                warnings.append(f"{ann_path}: missing image file on disk: {split_dir / fn}")

    seg_nonempty_count = 0
    seg_invalid_count = 0
    seg_valid_mask_anns_by_image: Dict[int, int] = {}
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
            if _segmentation_is_valid(seg):
                seg_valid_mask_anns_by_image[iid] = seg_valid_mask_anns_by_image.get(iid, 0) + 1
            else:
                seg_invalid_count += 1

    if task == "seg":
        # Background-only segmentation images are allowed, but still worth surfacing
        # because they contribute no mask supervision and older RF-DETR/Albumentations
        # stacks could crash on empty mask lists.
        empty_mask_images: List[str] = []
        for iid in image_ids:
            if seg_valid_mask_anns_by_image.get(iid, 0) <= 0:
                name = image_id_to_name.get(iid, f"id={iid}")
                empty_mask_images.append(str(name))

        if empty_mask_images:
            preview = ", ".join(empty_mask_images[:10]) + ("..." if len(empty_mask_images) > 10 else "")
            warnings.append(
                f"{ann_path}: task=seg but {len(empty_mask_images)} images have 0 valid masks "
                f"(example: {preview}). Background-only images are allowed, but they do not contribute "
                "mask supervision."
            )

    if task == "seg" and len(anns) > 0 and seg_nonempty_count == 0:
        errors.append(f"{ann_path}: task=seg but 0 annotations have non-empty segmentation (this is a detection-only dataset)")
    if task == "seg" and seg_invalid_count > 0:
        errors.append(f"{ann_path}: task=seg but {seg_invalid_count} annotations have invalid/empty segmentation")

    return CocoValidation(len(errors) == 0, errors, warnings)


def prune_empty_masks_in_split(split_dir: Path, *, dry_run: bool = False) -> CocoPruneResult:
    """For COCO segmentation, drop annotations with invalid/empty segmentation and drop images that end up with 0 masks.

    This does not delete any image files from disk; it only rewrites `_annotations.coco.json`.
    """
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return CocoPruneResult(False, message=f"Missing: {ann_path}")

    try:
        coco = load_json_strict(ann_path)
    except Exception as e:
        return CocoPruneResult(False, message=f"Failed to parse {ann_path}: {e}")

    images = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []
    if not isinstance(images, list) or not isinstance(anns, list):
        return CocoPruneResult(False, message=f"Invalid COCO structure in {ann_path} (images/annotations must be lists)")

    kept_anns: List[Dict[str, Any]] = []
    mask_ann_count: Dict[int, int] = {}
    removed_annotations = 0
    for a in anns:
        if not isinstance(a, dict):
            removed_annotations += 1
            continue
        seg = a.get("segmentation", None)
        if not _segmentation_is_valid(seg):
            removed_annotations += 1
            continue
        kept_anns.append(a)
        try:
            iid = int(a.get("image_id"))
            mask_ann_count[iid] = mask_ann_count.get(iid, 0) + 1
        except Exception:
            # keep it; validator can catch weird ids later
            pass

    kept_images: List[Dict[str, Any]] = []
    kept_image_ids: Set[int] = set()
    removed_images = 0
    for im in images:
        if not isinstance(im, dict):
            removed_images += 1
            continue
        try:
            iid = int(im.get("id"))
        except Exception:
            removed_images += 1
            continue
        if mask_ann_count.get(iid, 0) <= 0:
            removed_images += 1
            continue
        kept_images.append(im)
        kept_image_ids.add(iid)

    # Drop any annotations that reference images we dropped.
    kept_anns2: List[Dict[str, Any]] = []
    for a in kept_anns:
        try:
            iid = int(a.get("image_id"))
        except Exception:
            removed_annotations += 1
            continue
        if iid not in kept_image_ids:
            removed_annotations += 1
            continue
        kept_anns2.append(a)

    if dry_run:
        return CocoPruneResult(
            True,
            removed_images=int(removed_images),
            removed_annotations=int(removed_annotations),
            backup_path=None,
            message=f"{split_dir.name}: would remove {removed_images} images and {removed_annotations} annotations (dry-run)",
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = ann_path.with_name(f"_annotations.coco.backup_{ts}.json")
    try:
        shutil.copy2(str(ann_path), str(backup_path))
    except Exception:
        backup_path = None

    coco["images"] = kept_images
    coco["annotations"] = kept_anns2
    save_json(ann_path, coco)

    return CocoPruneResult(
        True,
        removed_images=int(removed_images),
        removed_annotations=int(removed_annotations),
        backup_path=backup_path,
        message=f"{split_dir.name}: removed {removed_images} images and {removed_annotations} annotations",
    )


def _poly_area(poly: List[float]) -> float:
    # Shoelace formula. poly = [x1,y1,x2,y2,...]
    if len(poly) < 6 or (len(poly) % 2) != 0:
        return 0.0
    xs = poly[0::2]
    ys = poly[1::2]
    n = len(xs)
    acc = 0.0
    for i in range(n):
        j = (i + 1) % n
        acc += float(xs[i]) * float(ys[j]) - float(xs[j]) * float(ys[i])
    return abs(acc) * 0.5


def prune_too_small_masks_in_split(
    split_dir: Path,
    *,
    resolution: int,
    min_scaled_area: float = 1.0,
    dry_run: bool = False,
) -> CocoPruneResult:
    """Drop seg annotations likely to vanish at a given square-resize resolution.

    RF-DETR's seg dataloader can crash with `masks cannot be empty` if, after resizing, an image ends up with 0 masks.
    This utility removes tiny instances (by scaled area heuristic) and then removes images with 0 remaining masks.
    """
    if resolution <= 0:
        return CocoPruneResult(False, message="resolution must be > 0")

    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return CocoPruneResult(False, message=f"Missing: {ann_path}")

    try:
        coco = load_json_strict(ann_path)
    except Exception as e:
        return CocoPruneResult(False, message=f"Failed to parse {ann_path}: {e}")

    images = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []
    if not isinstance(images, list) or not isinstance(anns, list):
        return CocoPruneResult(False, message=f"Invalid COCO structure in {ann_path} (images/annotations must be lists)")

    id_to_hw: Dict[int, Tuple[int, int]] = {}
    for im in images:
        if not isinstance(im, dict):
            continue
        try:
            iid = int(im.get("id"))
            w = int(im.get("width") or 0)
            h = int(im.get("height") or 0)
        except Exception:
            continue
        if w > 0 and h > 0:
            id_to_hw[iid] = (w, h)

    kept_anns: List[Dict[str, Any]] = []
    removed_annotations = 0
    kept_mask_count: Dict[int, int] = {}
    for a in anns:
        if not isinstance(a, dict):
            removed_annotations += 1
            continue
        seg = a.get("segmentation", None)
        if not _segmentation_is_valid(seg):
            removed_annotations += 1
            continue
        try:
            iid = int(a.get("image_id"))
        except Exception:
            removed_annotations += 1
            continue

        hw = id_to_hw.get(iid)
        if hw is None:
            # Can't scale; keep to avoid accidental data loss.
            kept_anns.append(a)
            kept_mask_count[iid] = kept_mask_count.get(iid, 0) + 1
            continue
        w, h = hw
        scale = float(resolution) / float(max(w, h))

        area = a.get("area", None)
        area_f = None
        if isinstance(area, (int, float)) and float(area) > 0:
            area_f = float(area)
        elif isinstance(seg, list):
            area_f = float(sum(_poly_area(poly) for poly in seg if isinstance(poly, list)))

        if area_f is not None:
            scaled = area_f * (scale * scale)
            if scaled < float(min_scaled_area):
                removed_annotations += 1
                continue

        kept_anns.append(a)
        kept_mask_count[iid] = kept_mask_count.get(iid, 0) + 1

    kept_images: List[Dict[str, Any]] = []
    kept_image_ids: Set[int] = set()
    removed_images = 0
    for im in images:
        if not isinstance(im, dict):
            removed_images += 1
            continue
        try:
            iid = int(im.get("id"))
        except Exception:
            removed_images += 1
            continue
        if kept_mask_count.get(iid, 0) <= 0:
            removed_images += 1
            continue
        kept_images.append(im)
        kept_image_ids.add(iid)

    kept_anns2: List[Dict[str, Any]] = []
    for a in kept_anns:
        try:
            iid = int(a.get("image_id"))
        except Exception:
            removed_annotations += 1
            continue
        if iid not in kept_image_ids:
            removed_annotations += 1
            continue
        kept_anns2.append(a)

    if dry_run:
        return CocoPruneResult(
            True,
            removed_images=int(removed_images),
            removed_annotations=int(removed_annotations),
            backup_path=None,
            message=(
                f"{split_dir.name}: would remove {removed_images} images and {removed_annotations} annotations "
                f"(resolution={resolution}, min_scaled_area={min_scaled_area}, dry-run)"
            ),
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = ann_path.with_name(f"_annotations.coco.backup_{ts}.json")
    try:
        shutil.copy2(str(ann_path), str(backup_path))
    except Exception:
        backup_path = None

    coco["images"] = kept_images
    coco["annotations"] = kept_anns2
    save_json(ann_path, coco)

    return CocoPruneResult(
        True,
        removed_images=int(removed_images),
        removed_annotations=int(removed_annotations),
        backup_path=backup_path,
        message=(
            f"{split_dir.name}: removed {removed_images} images and {removed_annotations} annotations "
            f"(resolution={resolution}, min_scaled_area={min_scaled_area})"
        ),
    )


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
            categories = load_json_strict(candidate).get("categories", []) or []
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
    save_json(ann_path, minimal)
    return ann_path


def _write_empty_split(ann_path: Path, *, categories: List[Dict[str, Any]]) -> None:
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    minimal = {
        "info": {"description": "empty split", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    save_json(ann_path, minimal)


def reset_coco_dir(dataset_dir: Path, *, backup: bool = True) -> Tuple[bool, str]:
    """Reset `<dataset>/coco` to empty train/valid/test splits (with categories from METADATA.json if present).

    This is useful when you ingested with the wrong label type (e.g. YOLO polygons but used detect),
    or when you want to start fresh without creating a new UUID.
    """
    dataset_dir = resolve_path(dataset_dir)
    coco_dir = dataset_dir / "coco"
    md_path = dataset_dir / "METADATA.json"

    class_names: List[str] = []
    if md_path.exists():
        try:
            md = load_json_strict(md_path)
            class_names = list(md.get("class_names", []) or [])
        except Exception:
            class_names = []

    categories = [{"id": i, "name": n, "supercategory": ""} for i, n in enumerate(class_names)] if class_names else []

    if coco_dir.exists() and backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = dataset_dir / f"coco_backup_{ts}"
        shutil.move(str(coco_dir), str(bak))
    else:
        coco_dir.mkdir(parents=True, exist_ok=True)

    for sp in ("train", "valid", "test"):
        _write_empty_split(coco_dir / sp / "_annotations.coco.json", categories=categories)
    return True, f"Reset COCO splits at {coco_dir} (backup={'yes' if backup else 'no'})"


def normalize_coco_category_ids(ann_path: Path, *, dry_run: bool = False) -> Tuple[bool, str]:
    """Rewrite categories/annotations to use contiguous 0..N-1 category ids (in ascending old-id order).

    Creates a `.bak` file next to the JSON (unless dry_run=True).
    """
    if not ann_path.exists():
        return False, f"Not found: {ann_path}"

    coco = load_json_strict(ann_path)
    cats = coco.get("categories", []) or []
    anns = coco.get("annotations", []) or []
    if not isinstance(cats, list) or not isinstance(anns, list):
        return False, f"Invalid COCO structure in {ann_path} (categories/annotations must be lists)"

    old_ids: List[int] = []
    id_to_cat: Dict[int, Dict[str, Any]] = {}
    for c in cats:
        try:
            cid = int(c.get("id"))
        except Exception:
            continue
        if cid not in id_to_cat:
            id_to_cat[cid] = c
            old_ids.append(cid)

    if not old_ids:
        return False, f"No categories found in {ann_path}"

    old_ids_sorted = sorted(old_ids)
    mapping = {old: new for new, old in enumerate(old_ids_sorted)}

    # If already normalized, do nothing
    if old_ids_sorted == list(range(0, len(old_ids_sorted))):
        return True, f"Already normalized: {ann_path}"

    new_cats: List[Dict[str, Any]] = []
    for old in old_ids_sorted:
        c = dict(id_to_cat[old])
        c["id"] = mapping[old]
        new_cats.append(c)

    new_anns: List[Dict[str, Any]] = []
    for a in anns:
        try:
            old = int(a.get("category_id"))
        except Exception:
            new_anns.append(a)
            continue
        if old in mapping:
            aa = dict(a)
            aa["category_id"] = mapping[old]
            new_anns.append(aa)
        else:
            new_anns.append(a)

    coco["categories"] = new_cats
    coco["annotations"] = new_anns

    if dry_run:
        return True, f"Would normalize ids in: {ann_path}"

    bak = ann_path.with_suffix(ann_path.suffix + ".bak")
    if not bak.exists():
        save_json(bak, load_json_strict(ann_path))

    save_json(ann_path, coco)
    return True, f"Normalized category ids in: {ann_path}"


def patch_coco_categories_supercategory(ann_path: Path, *, default: str = "") -> Tuple[bool, str]:
    """Ensure every categories[] entry has a 'supercategory' key (RF-DETR may assume it exists).

    Writes a `.bak` file next to the JSON if changes are made.
    Returns (changed, message).
    """
    if not ann_path.exists():
        return False, f"Not found: {ann_path}"

    try:
        coco = load_json_strict(ann_path)
    except Exception as e:
        return False, f"Failed to parse {ann_path}: {e}"

    cats = coco.get("categories", None)
    if not isinstance(cats, list):
        return False, f"{ann_path}: categories is not a list"

    changed = False
    new_cats: List[Dict[str, Any]] = []
    for c in cats:
        if not isinstance(c, dict):
            new_cats.append({"id": c, "name": str(c), "supercategory": default})
            changed = True
            continue
        cc = dict(c)
        if "supercategory" not in cc:
            cc["supercategory"] = default
            changed = True
        new_cats.append(cc)

    if not changed:
        return False, f"OK: {ann_path}"

    bak = ann_path.with_suffix(ann_path.suffix + ".bak")
    if not bak.exists():
        save_json(bak, load_json_strict(ann_path))

    coco["categories"] = new_cats
    save_json(ann_path, coco)
    return True, f"Patched categories.supercategory in: {ann_path}"


def align_coco_categories_to_metadata(ann_path: Path, *, class_names: List[str], dry_run: bool = False) -> Tuple[bool, str]:
    """Force categories[] to match METADATA.json class_names and remap annotation category_id by category name.

    This prevents duplicated class names (and wrong num_classes) when merging COCO sources.
    """
    if not ann_path.exists():
        return False, f"Not found: {ann_path}"
    if not class_names:
        return False, "No class_names provided"

    try:
        coco = load_json_strict(ann_path)
    except Exception as e:
        return False, f"Failed to parse {ann_path}: {e}"

    cats = coco.get("categories", []) or []
    anns = coco.get("annotations", []) or []
    if not isinstance(cats, list) or not isinstance(anns, list):
        return False, f"Invalid COCO structure in {ann_path}"

    # old category_id -> category_name
    old_id_to_name: Dict[int, str] = {}
    for c in cats:
        if not isinstance(c, dict):
            continue
        try:
            cid = int(c.get("id"))
        except Exception:
            continue
        name = str(c.get("name", "")).strip()
        if name:
            old_id_to_name[cid] = name

    name_to_new = {str(n): i for i, n in enumerate(class_names)}

    remapped = 0
    unknown = 0
    for a in anns:
        try:
            old = int(a.get("category_id"))
        except Exception:
            continue
        name = old_id_to_name.get(old, "")
        if not name:
            unknown += 1
            continue
        if name not in name_to_new:
            unknown += 1
            continue
        new = int(name_to_new[name])
        if new != old:
            if not dry_run:
                a["category_id"] = new
            remapped += 1

    if unknown:
        return False, f"{ann_path}: found {unknown} annotations with category ids/names not in METADATA.json class_names"

    new_categories = [{"id": i, "name": n, "supercategory": ""} for i, n in enumerate(class_names)]
    if dry_run:
        return True, f"Would align categories to METADATA.json for: {ann_path} (remapped {remapped} annotations)"

    bak = ann_path.with_suffix(ann_path.suffix + ".bak")
    if not bak.exists():
        save_json(bak, load_json_strict(ann_path))

    coco["categories"] = new_categories
    coco["annotations"] = anns
    save_json(ann_path, coco)
    return True, f"Aligned categories to METADATA.json for: {ann_path} (remapped {remapped} annotations)"


def subsample_coco_split(
    split_dir: Path,
    *,
    fraction: Optional[float] = None,
    max_images: Optional[int] = None,
    min_instances_per_class: int = 1,
    seed: int = 42,
    dry_run: bool = False
) -> CocoPruneResult:
    """Subsample a COCO split ensuring class representation and proportional background images."""
    import random
    from collections import defaultdict

    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return CocoPruneResult(False, message=f"Missing: {ann_path}")

    try:
        coco = load_json_strict(ann_path)
    except Exception as e:
        return CocoPruneResult(False, message=f"Failed to parse {ann_path}: {e}")

    images = coco.get("images", []) or []
    anns = coco.get("annotations", []) or []
    cats = coco.get("categories", []) or []
    
    if not images:
        return CocoPruneResult(False, message=f"No images found in {ann_path}")

    total_imgs = len(images)
    
    if max_images is not None:
        target_count = max(1, min(max_images, total_imgs))
    elif fraction is not None:
        target_count = max(1, min(int(total_imgs * fraction), total_imgs))
    else:
        return CocoPruneResult(False, message="Must specify either fraction or max_images")

    if target_count == total_imgs:
        return CocoPruneResult(True, message="Target count equals total images, no subsampling needed.")

    rng = random.Random(seed)

    # 1. Initialization & Profiling
    img_id_to_cats = defaultdict(set)
    img_id_to_ann_count = defaultdict(int)
    for a in anns:
        try:
            iid = int(a.get("image_id"))
            cid = int(a.get("category_id"))
            img_id_to_cats[iid].add(cid)
            img_id_to_ann_count[iid] += 1
        except Exception:
            continue

    all_image_ids = set()
    for im in images:
        try:
            all_image_ids.add(int(im.get("id")))
        except Exception:
            pass

    background_img_ids = [iid for iid in all_image_ids if img_id_to_ann_count.get(iid, 0) == 0]
    labeled_img_ids = [iid for iid in all_image_ids if img_id_to_ann_count.get(iid, 0) > 0]

    # Calculate frequencies
    cat_freq = defaultdict(int)
    cat_to_imgs = defaultdict(list)
    for iid in labeled_img_ids:
        for cid in img_id_to_cats[iid]:
            cat_freq[cid] += 1
            cat_to_imgs[cid].append(iid)

    # Sort categories by rarity
    sorted_cats = sorted(cat_freq.keys(), key=lambda c: cat_freq[c])

    selected_image_ids: Set[int] = set()

    # 2. Phase 1: Class Guarantee (Stratified Greedy Selection)
    for cid in sorted_cats:
        # Check current representation of this class
        current_count = sum(1 for iid in selected_image_ids if cid in img_id_to_cats[iid])
        
        if current_count < min_instances_per_class:
            needed = min_instances_per_class - current_count
            available = [iid for iid in cat_to_imgs[cid] if iid not in selected_image_ids]
            # Shuffle to avoid always picking the same images for a class
            rng.shuffle(available)
            for iid in available[:needed]:
                selected_image_ids.add(iid)

    # Calculate remaining slots after class guarantee
    # Note: Phase 1 might exceed target_count for extremely small subsets, 
    # but we prioritize class guarantee over absolute subset size.

    # 3. Phase 2: Background Proportionality
    orig_bg_ratio = len(background_img_ids) / total_imgs if total_imgs > 0 else 0.0
    target_bg_count = int(target_count * orig_bg_ratio)
    
    current_bg_count = len(selected_image_ids.intersection(set(background_img_ids)))
    bg_needed = max(0, target_bg_count - current_bg_count)
    
    available_bg = [iid for iid in background_img_ids if iid not in selected_image_ids]
    rng.shuffle(available_bg)
    
    for iid in available_bg[:bg_needed]:
        selected_image_ids.add(iid)

    # 4. Phase 3: Random Fill
    slots_left = target_count - len(selected_image_ids)
    if slots_left > 0:
        available_any = [iid for iid in all_image_ids if iid not in selected_image_ids]
        rng.shuffle(available_any)
        for iid in available_any[:slots_left]:
            selected_image_ids.add(iid)

    # 5. Construct & Save
    kept_images = [im for im in images if int(im.get("id", -1)) in selected_image_ids]
    kept_anns = [a for a in anns if int(a.get("image_id", -1)) in selected_image_ids]

    removed_images = len(images) - len(kept_images)
    removed_annotations = len(anns) - len(kept_anns)

    if dry_run:
        return CocoPruneResult(
            True,
            removed_images=removed_images,
            removed_annotations=removed_annotations,
            backup_path=None,
            message=(
                f"{split_dir.name}: would keep {len(kept_images)}/{total_imgs} images "
                f"and {len(kept_anns)}/{len(anns)} annotations (dry-run)"
            ),
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = ann_path.with_name(f"_annotations.coco.backup_subsample_{ts}.json")
    try:
        shutil.copy2(str(ann_path), str(backup_path))
    except Exception:
        backup_path = None

    coco["images"] = kept_images
    coco["annotations"] = kept_anns
    save_json(ann_path, coco)

    return CocoPruneResult(
        True,
        removed_images=removed_images,
        removed_annotations=removed_annotations,
        backup_path=backup_path,
        message=(
            f"{split_dir.name}: subsampled to {len(kept_images)} images "
            f"(-{removed_images} images, -{removed_annotations} annotations)"
        ),
    )

