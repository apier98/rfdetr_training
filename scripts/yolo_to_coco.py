#!/usr/bin/env python3
"""
Convert YOLO-format labels to COCO JSON and split images into train/valid.

Supports two tasks:
- detect: YOLO bbox labels  -> COCO bbox (segmentation empty)
- seg:    YOLO polygon labels -> COCO instance segmentation polygons + bbox + area

Usage examples:
  # Detection (bbox labels: class x_center y_center width height)
  python scripts/yolo_to_coco.py --dataset-dir datasets/<UUID> --task detect --train-ratio 0.8

  # Segmentation (polygon labels: class x1 y1 x2 y2 ... ; coords normalized [0..1])
  python scripts/yolo_to_coco.py --dataset-dir datasets/<UUID> --task seg --train-ratio 0.8

Behavior:
- Reads `METADATA.json` from the dataset root to obtain `class_names`.
- Reads images from `<dataset>/raw/` and YOLO labels from `<dataset>/yolo/`.
- Produces:
    coco/train/_annotations.coco.json
    coco/valid/_annotations.coco.json
  and optionally copies images into corresponding folders.

Notes:
- Class ids in YOLO are preserved (expected 0-based). Categories in COCO will use the same ids when class_names exist.
- Only train/valid splits are created (no test split).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert YOLO labels to COCO format and split into train/valid")
    p.add_argument(
        "--dataset-dir",
        "-d",
        required=True,
        help="Path to dataset UUID folder (contains raw/, yolo/, METADATA.json)",
    )
    p.add_argument(
        "--task",
        choices=["detect", "seg"],
        default="detect",
        help="Conversion mode: detect (bbox) or seg (polygon instance segmentation). Default: detect",
    )
    p.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of images to use for training (default 0.8)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for shuffling/splitting")
    p.add_argument("--copy-images", action="store_true", help="Copy images into coco/train and coco/valid (default: no)")
    p.add_argument("--images-ext", default="jpg,png", help="Comma-separated image extensions to consider (default: jpg,png)")
    p.add_argument("--validate", action="store_true", help="Print a validation report (continues with conversion)")
    p.add_argument("--validate-only", action="store_true", help="Run validation report and do NOT write COCO files or copy images")
    return p.parse_args()


# ---------------------------
# Helpers
# ---------------------------

def load_metadata(dataset_dir: Path) -> Dict[str, Any]:
    md_path = dataset_dir / "METADATA.json"
    if not md_path.exists():
        print(f"Warning: METADATA.json not found at {md_path}. Categories may be inferred from labels.")
        return {}
    try:
        return json.loads(md_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: failed to parse METADATA.json ({e}). Proceeding without class_names.")
        return {}


def find_images(raw_dir: Path, exts: List[str]) -> List[Path]:
    out: List[Path] = []
    for ext in exts:
        out.extend(sorted(raw_dir.rglob(f"*.{ext}")))
        out.extend(sorted(raw_dir.rglob(f"*.{ext.upper()}")))
    # de-dup
    seen = set()
    uniq: List[Path] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def image_size(img_path: Path) -> Tuple[int, int]:
    # Avoid heavy deps; use PIL if available
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        raise ImportError("Pillow is required to read image sizes. Install with: pip install pillow")
    with Image.open(img_path) as im:
        return int(im.width), int(im.height)


def read_yolo_labels_detect(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO detect label format:
      class x_center y_center width height
    All coords normalized to [0..1].
    """
    if not label_path.exists():
        return []
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: List[Tuple[int, float, float, float, float]] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            vals = [float(x) for x in parts[1:5]]
            out.append((cls, vals[0], vals[1], vals[2], vals[3]))
        except Exception:
            continue
    return out


def read_yolo_labels_seg(label_path: Path) -> List[Tuple[int, List[float]]]:
    """
    YOLO segmentation (polygon) label format (common):
      class x1 y1 x2 y2 x3 y3 ...
    All coords normalized to [0..1].

    Returns list of (class_id, coords_norm_flat) where coords_norm_flat = [x1,y1,x2,y2,...]
    """
    if not label_path.exists():
        return []
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: List[Tuple[int, List[float]]] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 7:  # class + at least 3 points (6 numbers)
            continue
        try:
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            # must be even number of coords
            if len(coords) % 2 != 0:
                continue
            # at least 3 points
            if len(coords) < 6:
                continue
            out.append((cls, coords))
        except Exception:
            continue
    return out


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def norm_poly_to_pixel(coords_norm_flat: List[float], w: int, h: int) -> List[float]:
    """Convert normalized polygon coords to pixel coords, clamped to image bounds."""
    px: List[float] = []
    for i, v in enumerate(coords_norm_flat):
        if i % 2 == 0:
            # x
            x = v * w
            x = clamp(x, 0.0, float(w - 1))
            px.append(float(x))
        else:
            # y
            y = v * h
            y = clamp(y, 0.0, float(h - 1))
            px.append(float(y))
    return px


def bbox_from_poly(px_flat: List[float]) -> Tuple[float, float, float, float]:
    xs = px_flat[0::2]
    ys = px_flat[1::2]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min)


def polygon_area(px_flat: List[float]) -> float:
    """Shoelace formula for polygon area. Assumes a simple (non-self-intersecting) polygon."""
    xs = px_flat[0::2]
    ys = px_flat[1::2]
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area2) / 2.0


def coco_dict(images_info: List[Dict], annotations: List[Dict], categories: List[Dict]) -> Dict:
    info = {"description": "Converted from YOLO", "version": "1.0"}
    return {
        "info": info,
        "licenses": [],
        "images": images_info,
        "annotations": annotations,
        "categories": categories,
    }


# ---------------------------
# Conversion
# ---------------------------

def convert(
    dataset_dir: Path,
    task: str,
    train_ratio: float,
    seed: int,
    copy_images: bool,
    exts: List[str],
    validate: bool = False,
    validate_only: bool = False,
) -> None:
    raw_dir = dataset_dir / "raw"
    yolo_dir = dataset_dir / "yolo"
    coco_dir = dataset_dir / "coco"

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw/ directory not found at {raw_dir}")
    if not yolo_dir.exists():
        print(f"Warning: yolo/ directory not found at {yolo_dir}. Proceeding with empty labels.")

    metadata = load_metadata(dataset_dir)
    class_names = metadata.get("class_names", []) or []
    if not class_names:
        print("Warning: METADATA.json has empty class_names. YOLO numeric ids will be used as category ids.")

    imgs = find_images(raw_dir, exts)
    if not imgs:
        raise RuntimeError(f"No images found under {raw_dir} with extensions {exts}")

    # map by stem
    basename_to_path: Dict[str, Path] = {p.stem: p for p in imgs}

    labels = list(yolo_dir.glob("*.txt")) if yolo_dir.exists() else []
    # validation stats: max class id seen
    max_label_id = -1
    for lab in labels:
        for ln in lab.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if not parts:
                continue
            try:
                cls_id = int(parts[0])
                if cls_id > max_label_id:
                    max_label_id = cls_id
            except Exception:
                continue

    if validate or validate_only:
        print("Validation report for dataset:")
        print(f"  task: {task}")
        print(f"  images found under raw/: {len(imgs)}")
        print(f"  label files found under yolo/: {len(labels)}")
        print(f"  max class id found in labels: {max_label_id}")
        if class_names:
            print(f"  class_names count (from METADATA.json): {len(class_names)}")
            if max_label_id >= len(class_names):
                print(
                    f"  WARNING: max label id {max_label_id} >= class_names length {len(class_names)}. "
                    "Your METADATA.json may be missing classes or labels are out of range."
                )
        else:
            print("  class_names: (none)")

        if validate_only:
            print("--validate-only set; exiting after validation without writing COCO files.")
            return

    # images to include: those in raw/
    image_stems = list(basename_to_path.keys())
    random.Random(seed).shuffle(image_stems)
    cut = int(len(image_stems) * train_ratio)
    train_stems = set(image_stems[:cut])
    valid_stems = set(image_stems[cut:])

    # categories
    categories: List[Dict[str, Any]] = []
    for idx, name in enumerate(class_names):
        categories.append({"id": idx, "name": name, "supercategory": ""})

    # fallback if class_names empty -> dynamic map from labels
    dynamic_categories: Dict[int, int] = {}

    images_info_train: List[Dict[str, Any]] = []
    images_info_valid: List[Dict[str, Any]] = []
    annotations_train: List[Dict[str, Any]] = []
    annotations_valid: List[Dict[str, Any]] = []

    img_id = 1
    ann_id = 1

    for stem in image_stems:
        img_path = basename_to_path[stem]
        w, h = image_size(img_path)

        image_entry = {"id": img_id, "file_name": img_path.name, "width": w, "height": h}

        # read labels
        label_path = yolo_dir / f"{stem}.txt"
        if task == "seg":
            yolo_objs_seg = read_yolo_labels_seg(label_path)
            yolo_objs_det = None
        else:
            yolo_objs_det = read_yolo_labels_detect(label_path)
            yolo_objs_seg = None

        anns_for_image: List[Dict[str, Any]] = []

        if task == "seg":
            assert yolo_objs_seg is not None
            for (cls, coords_norm_flat) in yolo_objs_seg:
                # category id mapping
                if class_names:
                    cat_id = int(cls)
                else:
                    if cls not in dynamic_categories:
                        dynamic_categories[cls] = len(dynamic_categories)
                    cat_id = dynamic_categories[cls]

                px_flat = norm_poly_to_pixel(coords_norm_flat, w, h)
                x_min, y_min, box_w, box_h = bbox_from_poly(px_flat)
                area = polygon_area(px_flat)

                # skip degenerate
                if box_w <= 0.0 or box_h <= 0.0 or area <= 0.0:
                    continue

                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x_min, y_min, box_w, box_h],
                    "area": area,
                    "iscrowd": 0,
                    # COCO expects list-of-polygons; we provide one polygon per instance
                    "segmentation": [px_flat],
                }
                anns_for_image.append(ann)
                ann_id += 1

        else:
            assert yolo_objs_det is not None
            for (cls, x_c, y_c, bw, bh) in yolo_objs_det:
                # bbox in pixels
                x_min = (x_c - bw / 2.0) * w
                y_min = (y_c - bh / 2.0) * h
                box_w = bw * w
                box_h = bh * h

                # clamp to image bounds (COCO prefers within image)
                x_min = clamp(x_min, 0.0, float(w - 1))
                y_min = clamp(y_min, 0.0, float(h - 1))
                box_w = clamp(box_w, 0.0, float(w))
                box_h = clamp(box_h, 0.0, float(h))

                area = float(box_w * box_h)
                if box_w <= 0.0 or box_h <= 0.0:
                    continue

                # category id mapping
                if class_names:
                    cat_id = int(cls)
                else:
                    if cls not in dynamic_categories:
                        dynamic_categories[cls] = len(dynamic_categories)
                    cat_id = dynamic_categories[cls]

                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [float(x_min), float(y_min), float(box_w), float(box_h)],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],  # no masks in detection mode
                }
                anns_for_image.append(ann)
                ann_id += 1

        # assign to split
        if stem in train_stems:
            images_info_train.append(image_entry)
            annotations_train.extend(anns_for_image)
        else:
            images_info_valid.append(image_entry)
            annotations_valid.extend(anns_for_image)

        img_id += 1

    # If dynamic categories were used, construct categories list from map
    if not class_names:
        inv = {v: k for k, v in dynamic_categories.items()}
        categories = []
        for new_id in sorted(inv.keys()):
            orig_cls = inv[new_id]
            categories.append({"id": int(new_id), "name": str(orig_cls), "supercategory": ""})

    # write output
    train_out_dir = coco_dir / "train"
    valid_out_dir = coco_dir / "valid"
    train_out_dir.mkdir(parents=True, exist_ok=True)
    valid_out_dir.mkdir(parents=True, exist_ok=True)

    train_json = coco_dict(images_info_train, annotations_train, categories)
    valid_json = coco_dict(images_info_valid, annotations_valid, categories)

    (train_out_dir / "_annotations.coco.json").write_text(json.dumps(train_json, indent=2), encoding="utf-8")
    (valid_out_dir / "_annotations.coco.json").write_text(json.dumps(valid_json, indent=2), encoding="utf-8")

    # copy images if requested
    if copy_images:
        for img in imgs:
            stem = img.stem
            if stem in train_stems:
                dst = train_out_dir / img.name
            else:
                dst = valid_out_dir / img.name
            if dst.exists():
                continue
            shutil.copy2(img, dst)

    print("Done.")
    print(f"  Task: {task}")
    print(f"  Train images: {len(images_info_train)} | Train annotations: {len(annotations_train)}")
    print(f"  Valid images: {len(images_info_valid)} | Valid annotations: {len(annotations_valid)}")
    print(f"  Wrote: {train_out_dir / '_annotations.coco.json'}")
    print(f"  Wrote: {valid_out_dir / '_annotations.coco.json'}")
    if copy_images:
        print("  Copied images into coco/train and coco/valid.")


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    exts = [e.strip().lower() for e in args.images_ext.split(",") if e.strip()]
    try:
        convert(
            dataset_dir=dataset_dir,
            task=args.task,
            train_ratio=args.train_ratio,
            seed=args.seed,
            copy_images=args.copy_images,
            exts=exts,
            validate=args.validate,
            validate_only=args.validate_only,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
