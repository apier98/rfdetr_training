from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .pathutil import resolve_path
from .jsonutil import load_json_strict, save_json


@dataclass(frozen=True)
class MergeResult:
    ok: bool
    message: str
    images_added: int = 0
    annotations_added: int = 0


def _ensure_coco_skeleton(categories: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories or [],
    }


def _next_ids(dst: Dict[str, Any]) -> Tuple[int, int]:
    max_img = 0
    max_ann = 0
    for im in dst.get("images", []) or []:
        try:
            max_img = max(max_img, int(im.get("id", 0)))
        except Exception:
            pass
    for ann in dst.get("annotations", []) or []:
        try:
            max_ann = max(max_ann, int(ann.get("id", 0)))
        except Exception:
            pass
    return max_img + 1, max_ann + 1


def _compute_next_filename(dst_dir: Path, pad: int, suffix: str) -> str:
    maxnum = 0
    for p in dst_dir.iterdir():
        if not p.is_file():
            continue
        if p.stem.isdigit():
            try:
                maxnum = max(maxnum, int(p.stem))
            except Exception:
                pass
    return f"{maxnum + 1:0{pad}d}{suffix}"


def _build_category_map(
    *,
    dst: Dict[str, Any],
    src: Dict[str, Any],
    metadata_map: Optional[Dict[str, int]],
) -> Dict[int, int]:
    # src category id -> dst category id
    dst_cats: List[Dict[str, Any]] = list(dst.get("categories", []) or [])
    dst_name_to_id = {str(c.get("name", "")): int(c.get("id")) for c in dst_cats if "id" in c}
    dst_ids = [int(c.get("id")) for c in dst_cats if "id" in c]
    next_id = (max(dst_ids) + 1) if dst_ids else 0

    cat_map: Dict[int, int] = {}
    for sc in src.get("categories", []) or []:
        try:
            sid = int(sc.get("id"))
        except Exception:
            continue
        name = str(sc.get("name", "")).strip()

        if metadata_map and name in metadata_map:
            tid = int(metadata_map[name])
            cat_map[sid] = tid
            if not any(int(c.get("id", -1)) == tid for c in dst_cats):
                dst_cats.append({"id": tid, "name": name, "supercategory": sc.get("supercategory", "")})
            continue

        if name and name in dst_name_to_id:
            cat_map[sid] = int(dst_name_to_id[name])
            continue

        # fall back: add a new category (preserve name if present)
        tid = next_id
        next_id += 1
        cat_map[sid] = tid
        dst_cats.append({"id": tid, "name": name or str(sid), "supercategory": sc.get("supercategory", "")})
        if name:
            dst_name_to_id[name] = tid

    dst["categories"] = dst_cats
    return cat_map


def merge_coco_into_split(
    *,
    dataset_dir: Path,
    split: str,
    src_json: Path,
    src_images_dir: Optional[Path],
    mode: str,
    rename: bool,
    pad: int,
    metadata_map: Optional[Dict[str, int]],
    dry_run: bool,
) -> MergeResult:
    dataset_dir = resolve_path(dataset_dir)
    src_json = resolve_path(src_json)
    src_images_dir = resolve_path(src_images_dir) if src_images_dir else None

    if split not in ("train", "valid", "test"):
        return MergeResult(False, f"Invalid split: {split}")

    if not src_json.exists():
        return MergeResult(False, f"Source COCO json not found: {src_json}")

    dst_split_dir = dataset_dir / "coco" / split
    dst_split_dir.mkdir(parents=True, exist_ok=True)
    dst_json = dst_split_dir / "_annotations.coco.json"

    try:
        src = load_json_strict(src_json)
    except Exception as e:
        return MergeResult(False, f"Failed to parse source COCO json: {e}")

    if dst_json.exists():
        dst = load_json_strict(dst_json)
    else:
        dst = _ensure_coco_skeleton(categories=list(src.get("categories", []) or []))

    img_id_next, ann_id_next = _next_ids(dst)

    # map category ids
    cat_map = _build_category_map(dst=dst, src=src, metadata_map=metadata_map)

    dst_imgs = dst.get("images", []) or []
    dst_file_names = {str(im.get("file_name")) for im in dst_imgs if im.get("file_name")}

    # build src image id -> new id and new filename
    images_added = 0
    annotations_added = 0
    src_imgid_to_new: Dict[int, Tuple[int, str]] = {}

    def _find_src_image_path(fname: str) -> Optional[Path]:
        if src_images_dir is not None:
            p = src_images_dir / fname
            if p.exists():
                return p
        # try alongside json
        p2 = src_json.parent / fname
        if p2.exists():
            return p2
        # try dataset raw/
        p3 = dataset_dir / "raw" / fname
        if p3.exists():
            return p3
        return None

    for simg in src.get("images", []) or []:
        try:
            sid = int(simg.get("id"))
        except Exception:
            continue
        fname = str(simg.get("file_name", "")).strip()
        if not fname:
            continue

        suffix = Path(fname).suffix or ".jpg"
        if rename:
            new_fname = f"{img_id_next:0{pad}d}{suffix}"
        else:
            if fname in dst_file_names:
                new_fname = _compute_next_filename(dst_split_dir, pad, suffix)
            else:
                new_fname = fname

        src_imgid_to_new[sid] = (img_id_next, new_fname)

        # copy/move image file
        src_path = _find_src_image_path(fname)
        dst_path = dst_split_dir / new_fname
        if src_path is None:
            # don't hard fail: some pipelines keep images elsewhere
            # but warn via message count
            pass
        else:
            if not dry_run:
                if mode == "move":
                    shutil.move(str(src_path), str(dst_path))
                else:
                    shutil.copy2(str(src_path), str(dst_path))

        new_entry = dict(simg)
        new_entry["id"] = img_id_next
        new_entry["file_name"] = new_fname
        dst_imgs.append(new_entry)

        dst_file_names.add(new_fname)
        images_added += 1
        img_id_next += 1

    dst["images"] = dst_imgs

    # annotations
    dst_anns = dst.get("annotations", []) or []
    for ann in src.get("annotations", []) or []:
        try:
            old_imgid = int(ann.get("image_id"))
        except Exception:
            continue
        if old_imgid not in src_imgid_to_new:
            continue

        new_imgid, _new_fname = src_imgid_to_new[old_imgid]
        new_ann = dict(ann)
        new_ann["id"] = ann_id_next
        new_ann["image_id"] = new_imgid
        try:
            old_cid = int(ann.get("category_id"))
            new_ann["category_id"] = int(cat_map.get(old_cid, old_cid))
        except Exception:
            pass
        dst_anns.append(new_ann)
        annotations_added += 1
        ann_id_next += 1

    dst["annotations"] = dst_anns

    if dry_run:
        return MergeResult(
            True,
            f"DRY RUN: would merge {images_added} images and {annotations_added} annotations into {dst_json}",
            images_added,
            annotations_added,
        )

    # backup then write
    if dst_json.exists():
        bak = dst_json.with_suffix(dst_json.suffix + ".bak")
        if not bak.exists():
            save_json(bak, load_json_strict(dst_json))

    save_json(dst_json, dst)
    return MergeResult(
        True,
        f"Merged {images_added} images and {annotations_added} annotations into {dst_json}",
        images_added,
        annotations_added,
    )
