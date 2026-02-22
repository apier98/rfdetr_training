from __future__ import annotations

import json
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    dataset_dir: Path
    uuid: str


def ensure_uuid(u: str) -> str:
    try:
        return str(_uuid.UUID(u))
    except Exception:
        raise ValueError(f"Invalid UUID: {u}")


def create_dataset(
    *,
    root: Path,
    uuid_str: Optional[str],
    name: Optional[str],
    force: bool,
    no_readme: bool,
    class_names: Optional[List[str]],
) -> DatasetLayout:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if uuid_str:
        uuid_final = ensure_uuid(uuid_str)
    else:
        uuid_final = str(_uuid.uuid4())

    dataset_dir = (root / uuid_final).resolve()
    if dataset_dir.exists() and not force:
        raise FileExistsError(f"Dataset directory already exists: {dataset_dir} (use --force to proceed)")

    subpaths = [
        "raw",
        Path("coco") / "train",
        Path("coco") / "valid",
        Path("coco") / "test",
        Path("staging") / "to_label",
        Path("staging") / "labeled",
        Path("staging") / "quarantine",
        "yolo",
        "models",
        "logs",
        "exports",
    ]
    for sp in subpaths:
        (dataset_dir / sp).mkdir(parents=True, exist_ok=True)

    metadata_path = dataset_dir / "METADATA.json"
    metadata = {
        "uuid": uuid_final,
        "name": name or "",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "class_names": class_names or [],
        "notes": "",
    }
    if not metadata_path.exists() or force:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not no_readme:
        readme_path = dataset_dir / "README.md"
        if not readme_path.exists() or force:
            txt = f"# Dataset {uuid_final}\n\n"
            if name:
                txt += f"Name: {name}\n\n"
            txt += (
                "COCO-format data should be placed under `coco/train`, `coco/valid`, and `coco/test`.\n\n"
                "Typical workflow:\n"
                "1) Put images in `raw/` and labels in `yolo/` (optional)\n"
                "2) Convert to COCO with the CLI (yolo->coco)\n"
                "3) Train with the CLI (train)\n"
            )
            readme_path.write_text(txt, encoding="utf-8")

    return DatasetLayout(root=root, dataset_dir=dataset_dir, uuid=uuid_final)


def load_metadata(dataset_dir: Path) -> Dict[str, Any]:
    md_path = dataset_dir / "METADATA.json"
    if not md_path.exists():
        return {}
    try:
        return json.loads(md_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_images(raw_dir: Path, exts: List[str]) -> List[Path]:
    out: List[Path] = []
    for ext in exts:
        out.extend(sorted(raw_dir.rglob(f"*.{ext}")))
        out.extend(sorted(raw_dir.rglob(f"*.{ext.upper()}")))
    seen = set()
    uniq: List[Path] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def image_size(img_path: Path) -> Tuple[int, int]:
    from PIL import Image  # type: ignore

    with Image.open(img_path) as im:
        return int(im.width), int(im.height)


def _read_yolo_labels_detect(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
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


def _read_yolo_labels_seg(label_path: Path) -> List[Tuple[int, List[float]]]:
    if not label_path.exists():
        return []
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: List[Tuple[int, List[float]]] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 7:
            continue
        try:
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if len(coords) % 2 != 0 or len(coords) < 6:
                continue
            out.append((cls, coords))
        except Exception:
            continue
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _norm_poly_to_pixel(coords_norm_flat: List[float], w: int, h: int) -> List[float]:
    px: List[float] = []
    for i, v in enumerate(coords_norm_flat):
        if i % 2 == 0:
            x = _clamp(v * w, 0.0, float(w - 1))
            px.append(float(x))
        else:
            y = _clamp(v * h, 0.0, float(h - 1))
            px.append(float(y))
    return px


def _bbox_from_poly(px_flat: List[float]) -> Tuple[float, float, float, float]:
    xs = px_flat[0::2]
    ys = px_flat[1::2]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min)


def _polygon_area(px_flat: List[float]) -> float:
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


def yolo_to_coco(
    *,
    dataset_dir: Path,
    task: str,
    train_ratio: float,
    seed: int,
    copy_images: bool,
    exts: List[str],
    validate: bool = False,
    validate_only: bool = False,
) -> None:
    import random
    import shutil

    dataset_dir = dataset_dir.expanduser().resolve()
    raw_dir = dataset_dir / "raw"
    yolo_dir = dataset_dir / "yolo"
    coco_dir = dataset_dir / "coco"

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw/ directory not found at {raw_dir}")
    if not yolo_dir.exists():
        if validate or validate_only:
            print(f"Warning: yolo/ directory not found at {yolo_dir}. Proceeding with empty labels.")

    metadata = load_metadata(dataset_dir)
    class_names = metadata.get("class_names", []) or []
    if validate or validate_only:
        if not class_names:
            print("Warning: METADATA.json has empty class_names. YOLO numeric ids will be used as category ids.")

    imgs = find_images(raw_dir, exts)
    if not imgs:
        raise RuntimeError(f"No images found under {raw_dir} with extensions {exts}")

    basename_to_path = {p.stem: p for p in imgs}
    image_stems = list(basename_to_path.keys())
    random.Random(seed).shuffle(image_stems)
    cut = int(len(image_stems) * train_ratio)
    train_stems = set(image_stems[:cut])

    labels = list(yolo_dir.glob("*.txt")) if yolo_dir.exists() else []
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
                max_label_id = max(max_label_id, int(parts[0]))
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
        if validate_only:
            print("--validate-only set; exiting after validation without writing COCO files.")
            return

    categories: List[Dict[str, Any]] = []
    for idx, name in enumerate(class_names):
        categories.append({"id": idx, "name": name, "supercategory": ""})
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

        label_path = yolo_dir / f"{stem}.txt"
        yolo_objs_seg = _read_yolo_labels_seg(label_path) if task == "seg" else None
        yolo_objs_det = _read_yolo_labels_detect(label_path) if task != "seg" else None

        anns_for_image: List[Dict[str, Any]] = []
        if task == "seg":
            assert yolo_objs_seg is not None
            for (cls, coords_norm_flat) in yolo_objs_seg:
                if class_names:
                    cat_id = int(cls)
                else:
                    if cls not in dynamic_categories:
                        dynamic_categories[cls] = len(dynamic_categories)
                    cat_id = dynamic_categories[cls]

                px_flat = _norm_poly_to_pixel(coords_norm_flat, w, h)
                x_min, y_min, box_w, box_h = _bbox_from_poly(px_flat)
                area = _polygon_area(px_flat)
                if box_w <= 0.0 or box_h <= 0.0 or area <= 0.0:
                    continue

                anns_for_image.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [x_min, y_min, box_w, box_h],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [px_flat],
                    }
                )
                ann_id += 1
        else:
            assert yolo_objs_det is not None
            for (cls, x_c, y_c, bw, bh) in yolo_objs_det:
                x_min = (x_c - bw / 2.0) * w
                y_min = (y_c - bh / 2.0) * h
                box_w = bw * w
                box_h = bh * h

                x_min = _clamp(x_min, 0.0, float(w - 1))
                y_min = _clamp(y_min, 0.0, float(h - 1))
                box_w = _clamp(box_w, 0.0, float(w))
                box_h = _clamp(box_h, 0.0, float(h))
                area = float(box_w * box_h)
                if box_w <= 0.0 or box_h <= 0.0:
                    continue

                if class_names:
                    cat_id = int(cls)
                else:
                    if cls not in dynamic_categories:
                        dynamic_categories[cls] = len(dynamic_categories)
                    cat_id = dynamic_categories[cls]

                anns_for_image.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [float(x_min), float(y_min), float(box_w), float(box_h)],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1

        if stem in train_stems:
            images_info_train.append(image_entry)
            annotations_train.extend(anns_for_image)
        else:
            images_info_valid.append(image_entry)
            annotations_valid.extend(anns_for_image)

        img_id += 1

    if not class_names:
        inv = {v: k for k, v in dynamic_categories.items()}
        categories = [{"id": int(new_id), "name": str(inv[new_id]), "supercategory": ""} for new_id in sorted(inv.keys())]

    def coco_dict(images_info: List[Dict], annotations: List[Dict], categories: List[Dict]) -> Dict:
        info = {"description": "Converted from YOLO", "version": "1.0"}
        return {"info": info, "licenses": [], "images": images_info, "annotations": annotations, "categories": categories}

    train_out_dir = coco_dir / "train"
    valid_out_dir = coco_dir / "valid"
    train_out_dir.mkdir(parents=True, exist_ok=True)
    valid_out_dir.mkdir(parents=True, exist_ok=True)

    (train_out_dir / "_annotations.coco.json").write_text(
        json.dumps(coco_dict(images_info_train, annotations_train, categories), indent=2), encoding="utf-8"
    )
    (valid_out_dir / "_annotations.coco.json").write_text(
        json.dumps(coco_dict(images_info_valid, annotations_valid, categories), indent=2), encoding="utf-8"
    )

    if copy_images:
        for img in imgs:
            dst = train_out_dir / img.name if img.stem in train_stems else valid_out_dir / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

    print("Done.")
    print(f"  Task: {task}")
    print(f"  Train images: {len(images_info_train)} | Train annotations: {len(annotations_train)}")
    print(f"  Valid images: {len(images_info_valid)} | Valid annotations: {len(annotations_valid)}")
    print(f"  Wrote: {train_out_dir / '_annotations.coco.json'}")
    print(f"  Wrote: {valid_out_dir / '_annotations.coco.json'}")
    if copy_images:
        print("  Copied images into coco/train and coco/valid.")
