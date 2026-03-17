from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .coco import align_coco_categories_to_metadata, patch_coco_categories_supercategory
from .coco_merge import merge_coco_into_split
from .datasets import load_metadata, yolo_to_coco


@dataclass(frozen=True)
class IngestResult:
    ok: bool
    message: str
    coco_jsons_processed: int = 0
    yolo_labels_processed: int = 0
    background_images_added: int = 0
    quarantined_items: int = 0
    train_images: int = 0
    train_annotations: int = 0
    valid_images: int = 0
    valid_annotations: int = 0


def _looks_like_coco(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    return all(k in data for k in ("images", "annotations", "categories"))


def _split_coco_by_ratio(src: Dict[str, Any], train_ratio: float, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    images = list(src.get("images", []) or [])
    anns = list(src.get("annotations", []) or [])
    cats = list(src.get("categories", []) or [])

    # index annotations by image_id
    ann_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in anns:
        try:
            iid = int(a.get("image_id"))
        except Exception:
            continue
        ann_by_img.setdefault(iid, []).append(a)

    ids = []
    for im in images:
        try:
            ids.append(int(im.get("id")))
        except Exception:
            continue

    rng = random.Random(seed)
    rng.shuffle(ids)
    cut = int(len(ids) * float(train_ratio))
    train_ids = set(ids[:cut])

    def _build(split_ids: Set[int]) -> Dict[str, Any]:
        out_images = []
        out_anns = []
        for im in images:
            try:
                iid = int(im.get("id"))
            except Exception:
                continue
            if iid not in split_ids:
                continue
            out_images.append(im)
            out_anns.extend(ann_by_img.get(iid, []))
        return {"info": src.get("info", {}), "licenses": src.get("licenses", []), "images": out_images, "annotations": out_anns, "categories": cats}

    train_json = _build(train_ids)
    valid_json = _build(set(ids) - train_ids)
    return train_json, valid_json


def _quarantine_write_json(quarantine_dir: Path, base_name: str, payload: Dict[str, Any]) -> Path:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    out = quarantine_dir / base_name
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _count_split_items(ann_path: Path) -> Tuple[int, int]:
    if not ann_path.exists():
        return 0, 0
    try:
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0
    images = payload.get("images", []) or []
    annotations = payload.get("annotations", []) or []
    return len(images), len(annotations)


def ingest_labels_inbox(
    *,
    dataset_dir: Path,
    train_ratio: float,
    seed: int,
    yolo_task: str,
    images_ext: List[str],
    mode: str,
    align_metadata: bool,
    include_background: bool,
    dry_run: bool,
) -> IngestResult:
    dataset_dir = dataset_dir.expanduser().resolve()
    inbox = dataset_dir / "labels_inbox"
    quarantine = inbox / "quarantine"

    if not inbox.exists():
        return IngestResult(False, f"labels_inbox folder not found: {inbox}")

    md = load_metadata(dataset_dir)
    class_names = md.get("class_names", []) or []
    metadata_map = {str(name): idx for idx, name in enumerate(class_names)} if (align_metadata and class_names) else None

    exports_tmp = dataset_dir / "exports" / "ingest_tmp"
    if not dry_run:
        exports_tmp.mkdir(parents=True, exist_ok=True)

    seen_file_names: Set[str] = set()
    quarantined = 0
    coco_processed = 0
    yolo_processed = 0
    background_added = 0

    # 1) Ingest COCO JSONs from inbox (excluding quarantine)
    coco_candidates = []
    for p in inbox.rglob("*.json"):
        if "quarantine" in p.parts:
            continue
        if _looks_like_coco(p):
            coco_candidates.append(p)
    coco_candidates = sorted(coco_candidates)

    for src_json_path in coco_candidates:
        try:
            src = json.loads(src_json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        images = src.get("images", []) or []
        anns = src.get("annotations", []) or []
        cats = src.get("categories", []) or []

        # detect conflicts by file_name (first wins, later quarantined)
        fname_to_imgid: Dict[str, int] = {}
        conflicts: Set[str] = set()
        for im in images:
            fname = str(im.get("file_name", "")).strip()
            if not fname:
                continue
            if fname in fname_to_imgid:
                conflicts.add(fname)
                continue
            fname_to_imgid[fname] = int(im.get("id", -1))
            if fname in seen_file_names:
                conflicts.add(fname)

        if conflicts:
            # quarantine conflicting subset from this JSON
            conflict_ids = {fname_to_imgid[f] for f in conflicts if f in fname_to_imgid}
            q_images = [im for im in images if int(im.get("id", -1)) in conflict_ids]
            q_anns = [a for a in anns if int(a.get("image_id", -1)) in conflict_ids]
            q_payload = {"info": src.get("info", {}), "licenses": src.get("licenses", []), "images": q_images, "annotations": q_anns, "categories": cats}

            if not dry_run:
                qname = f"{src_json_path.stem}.conflicts.json"
                _quarantine_write_json(quarantine, qname, q_payload)
            quarantined += len(conflicts)

            # remove conflicts from ingestion payload
            keep_images = [im for im in images if str(im.get("file_name", "")).strip() not in conflicts]
            keep_ids = {int(im.get("id")) for im in keep_images if "id" in im}
            keep_anns = [a for a in anns if int(a.get("image_id", -1)) in keep_ids]
            src = {"info": src.get("info", {}), "licenses": src.get("licenses", []), "images": keep_images, "annotations": keep_anns, "categories": cats}

        # mark seen
        for im in src.get("images", []) or []:
            fname = str(im.get("file_name", "")).strip()
            if fname:
                seen_file_names.add(fname)

        # split and merge
        train_json, valid_json = _split_coco_by_ratio(src, train_ratio=train_ratio, seed=seed)
        if not dry_run:
            tmp_dir = exports_tmp / "coco"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            train_tmp = tmp_dir / f"{src_json_path.stem}.train.json"
            valid_tmp = tmp_dir / f"{src_json_path.stem}.valid.json"
            train_tmp.write_text(json.dumps(train_json, indent=2), encoding="utf-8")
            valid_tmp.write_text(json.dumps(valid_json, indent=2), encoding="utf-8")

            merge_coco_into_split(
                dataset_dir=dataset_dir,
                split="train",
                src_json=train_tmp,
                src_images_dir=None,
                mode=mode,
                rename=False,
                pad=6,
                metadata_map=metadata_map,
                dry_run=False,
            )
            merge_coco_into_split(
                dataset_dir=dataset_dir,
                split="valid",
                src_json=valid_tmp,
                src_images_dir=None,
                mode=mode,
                rename=False,
                pad=6,
                metadata_map=metadata_map,
                dry_run=False,
            )

        coco_processed += 1

    # 2) Ingest YOLO label files from inbox/yolo (or inbox root *.txt)
    yolo_dir = inbox / "yolo"
    yolo_txts = list(yolo_dir.glob("*.txt")) if yolo_dir.exists() else []
    yolo_txts.extend([p for p in inbox.glob("*.txt") if p.is_file()])
    yolo_txts = sorted(set(yolo_txts))

    # map raw images by stem
    raw_dir = dataset_dir / "raw"
    raw_images = {}
    for ext in images_ext:
        for p in raw_dir.rglob(f"*.{ext}"):
            raw_images[p.stem] = p
        for p in raw_dir.rglob(f"*.{ext.upper()}"):
            raw_images[p.stem] = p

    tmp_yolo_dir = exports_tmp / "yolo_labels"
    tmp_coco_dir = exports_tmp / "yolo_to_coco"

    if not dry_run:
        tmp_yolo_dir.mkdir(parents=True, exist_ok=True)
        tmp_coco_dir.mkdir(parents=True, exist_ok=True)

    for txt in yolo_txts:
        stem = txt.stem
        img = raw_images.get(stem)
        if img is None:
            continue
        # conflict: already labeled via COCO
        if img.name in seen_file_names:
            if not dry_run:
                quarantine.mkdir(parents=True, exist_ok=True)
                shutil.move(str(txt), str(quarantine / txt.name))
            quarantined += 1
            continue
        # copy label into temp yolo dir
        if not dry_run:
            shutil.copy2(str(txt), str(tmp_yolo_dir / txt.name))
        seen_file_names.add(img.name)
        yolo_processed += 1

    if yolo_processed > 0 and not dry_run:
        yolo_to_coco(
            dataset_dir=dataset_dir,
            task=yolo_task,
            train_ratio=train_ratio,
            seed=seed,
            copy_images=True,
            exts=images_ext,
            validate=False,
            validate_only=False,
            out_dir=tmp_coco_dir,
            raw_dir=raw_dir,
            yolo_dir=tmp_yolo_dir,
            labeled_only=True,
            verbose=False,
        )

        # merge YOLO-generated COCO into splits
        merge_coco_into_split(
            dataset_dir=dataset_dir,
            split="train",
            src_json=tmp_coco_dir / "train" / "_annotations.coco.json",
            src_images_dir=(tmp_coco_dir / "train"),
            mode=mode,
            rename=False,
            pad=6,
            metadata_map=metadata_map,
            dry_run=False,
        )
        merge_coco_into_split(
            dataset_dir=dataset_dir,
            split="valid",
            src_json=tmp_coco_dir / "valid" / "_annotations.coco.json",
            src_images_dir=(tmp_coco_dir / "valid"),
            mode=mode,
            rename=False,
            pad=6,
            metadata_map=metadata_map,
            dry_run=False,
        )

    # 3) Include remaining raw images as background (empty annotations) if requested.
    if include_background:
        raw_dir = dataset_dir / "raw"
        raw_images: List[Path] = []
        for ext in images_ext:
            raw_images.extend(list(raw_dir.rglob(f"*.{ext}")))
            raw_images.extend(list(raw_dir.rglob(f"*.{ext.upper()}")))

        remaining = []
        for p in raw_images:
            if p.name not in seen_file_names:
                remaining.append(p)

        if remaining:
            rng = random.Random(seed)
            rng.shuffle(remaining)
            cut = int(len(remaining) * float(train_ratio))
            bg_train = remaining[:cut]
            bg_valid = remaining[cut:]

            # categories from metadata (ids 0..N-1)
            cats: List[Dict[str, Any]] = []
            if class_names:
                for idx, name in enumerate(class_names):
                    cats.append({"id": int(idx), "name": str(name), "supercategory": ""})

            def _make_bg_json(paths: List[Path]) -> Dict[str, Any]:
                images = []
                iid = 1
                for p in paths:
                    try:
                        from PIL import Image  # type: ignore

                        with Image.open(p) as im:
                            w, h = int(im.width), int(im.height)
                    except Exception:
                        w, h = 0, 0
                    images.append({"id": iid, "file_name": p.name, "width": w, "height": h})
                    iid += 1
                return {"info": {"description": "background images"}, "licenses": [], "images": images, "annotations": [], "categories": cats}

            if dry_run:
                background_added += len(remaining)
            else:
                bg_dir = exports_tmp / "background"
                bg_dir.mkdir(parents=True, exist_ok=True)
                train_json_p = bg_dir / "background.train.json"
                valid_json_p = bg_dir / "background.valid.json"
                train_json_p.write_text(json.dumps(_make_bg_json(bg_train), indent=2), encoding="utf-8")
                valid_json_p.write_text(json.dumps(_make_bg_json(bg_valid), indent=2), encoding="utf-8")

                merge_coco_into_split(
                    dataset_dir=dataset_dir,
                    split="train",
                    src_json=train_json_p,
                    src_images_dir=raw_dir,
                    mode=mode,
                    rename=False,
                    pad=6,
                    metadata_map=metadata_map,
                    dry_run=False,
                )
                merge_coco_into_split(
                    dataset_dir=dataset_dir,
                    split="valid",
                    src_json=valid_json_p,
                    src_images_dir=raw_dir,
                    mode=mode,
                    rename=False,
                    pad=6,
                    metadata_map=metadata_map,
                    dry_run=False,
                )
                background_added += len(remaining)

    # 4) Finalize: make COCO as compatible as possible with RF-DETR expectations
    # - ensure categories.supercategory exists
    # - align categories to METADATA.json to avoid duplicated names / wrong num_classes
    if not dry_run:
        coco_dir = dataset_dir / "coco"
        for sp in ("train", "valid", "test"):
            ann_path = coco_dir / sp / "_annotations.coco.json"
            patch_coco_categories_supercategory(ann_path, default="")
            if align_metadata and class_names:
                ok_align, _msg = align_coco_categories_to_metadata(ann_path, class_names=class_names, dry_run=False)
                if not ok_align:
                    # keep ingest non-fatal; training will surface it with a clear error
                    pass

    train_images = 0
    train_annotations = 0
    valid_images = 0
    valid_annotations = 0
    if not dry_run:
        train_images, train_annotations = _count_split_items(dataset_dir / "coco" / "train" / "_annotations.coco.json")
        valid_images, valid_annotations = _count_split_items(dataset_dir / "coco" / "valid" / "_annotations.coco.json")

    msg = (
        f"Ingest complete: task={yolo_task}, coco_jsons={coco_processed}, yolo_labels={yolo_processed}, "
        f"background_images={background_added}, quarantined={quarantined}. "
        f"Final split counts -> train: {train_images} images / {train_annotations} annotations, "
        f"valid: {valid_images} images / {valid_annotations} annotations. "
        f"Next: run `python -m rfdetr_training dataset validate -d {dataset_dir} --task {yolo_task}`"
    )
    return IngestResult(
        True,
        msg,
        coco_processed,
        yolo_processed,
        background_added,
        quarantined,
        train_images,
        train_annotations,
        valid_images,
        valid_annotations,
    )
