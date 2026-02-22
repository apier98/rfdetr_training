#!/usr/bin/env python3
"""Commit images and COCO JSONs from `staging/labeled/` into a dataset COCO split.

Features:
- Detect COCO JSONs in staging and merge images/annotations into `coco/<split>/_annotations.coco.json`.
- Move or copy image files from `staging/labeled/` into `coco/<split>/`.
- Optionally rename images to a numeric zero-padded sequence to avoid collisions.
- Supports `--dry-run` to preview actions.

Usage:
  python scripts/commit_from_staging.py --dataset datasets/<UUID> [--split train] [--mode move|copy] [--rename]

"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Commit staged labeled images/COCO into dataset split")
    p.add_argument("--dataset", "-d", required=True, help="Path to dataset UUID folder")
    p.add_argument("--split", choices=["train", "valid", "test"], default="train")
    p.add_argument("--split-distribution", help=(
        "Comma-separated split:percent pairs to distribute staged images across splits, "
        "e.g. 'train:80,valid:20'. If provided, --split is ignored."))
    p.add_argument("--assign-mode", choices=["random", "sequential"], default="random",
                   help="How to assign images to splits when using --split-distribution (default: random)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for deterministic split assignment")
    p.add_argument("--align-metadata", action="store_true", help="Align staged category ids to dataset METADATA.json class_names ordering when possible")
    p.add_argument("--mode", choices=["move", "copy"], default="move", help="Move or copy images from staging (default: move)")
    p.add_argument("--dry-run", action="store_true", help="Show planned actions without writing files")
    p.add_argument("--rename", action="store_true", help="Rename images to sequential zero-padded names to avoid collisions")
    p.add_argument("--pad", type=int, default=6, help="Zero-pad width when renaming (default: 6)")
    p.add_argument("--staging-subdir", default="staging/labeled", help="Relative staging folder (default: staging/labeled)")
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find_coco_jsons(staging: Path) -> List[Path]:
    return sorted([p for p in staging.iterdir() if p.is_file() and p.suffix.lower().endswith(".json")])


def ensure_target_coco(dataset_dir: Path, split: str, staged: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    target_path = dataset_dir / "coco" / split / "_annotations.coco.json"
    if target_path.exists():
        target = load_json(target_path)
    else:
        # create minimal COCO structure; try to infer categories from staged file
        target = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": []}
        staged_cats = staged.get("categories") or []
        if staged_cats:
            target["categories"] = staged_cats
    return target_path, target


def next_ids(target: Dict[str, Any]) -> Tuple[int, int]:
    max_img = 0
    max_ann = 0
    for img in target.get("images", []):
        try:
            max_img = max(max_img, int(img.get("id", 0)))
        except Exception:
            pass
    for ann in target.get("annotations", []):
        try:
            max_ann = max(max_ann, int(ann.get("id", 0)))
        except Exception:
            pass
    return max_img + 1, max_ann + 1


def build_cat_map(target: Dict[str, Any], staged: Dict[str, Any], metadata_map: Dict[str, int] | None = None) -> Dict[int, int]:
    # map staged category id -> target category id, adding new categories as needed
    name_to_id: Dict[str, int] = {c["name"]: int(c["id"]) for c in target.get("categories", [])}
    existing_ids = [int(c["id"]) for c in target.get("categories", [])]
    next_cat_id = max(existing_ids, default=-1) + 1
    cat_map: Dict[int, int] = {}

    for sc in staged.get("categories", []) or []:
        sid = int(sc.get("id", 0))
        sname = sc.get("name")
        # prefer metadata mapping when provided
        if metadata_map and sname in metadata_map:
            tid = int(metadata_map[sname])
            cat_map[sid] = tid
            # ensure target categories contains this mapping
            if not any(c.get("id") == tid and c.get("name") == sname for c in target.get("categories", [])):
                # remove any existing category with same name but different id
                target_cats = [c for c in target.get("categories", []) if c.get("name") != sname]
                target_cats.append({"id": tid, "name": sname})
                target["categories"] = target_cats
        else:
            if sname in name_to_id:
                cat_map[sid] = name_to_id[sname]
            else:
                # add category to target with next available id
                tid = next_cat_id
                next_cat_id += 1
                name_to_id[sname] = tid
                target.setdefault("categories", []).append({"id": tid, "name": sname})
                cat_map[sid] = tid
    return cat_map


def compute_next_filename(target_dir: Path, pad: int) -> str:
    # find max numeric basename in target_dir and return next name (jpg)
    maxnum = 0
    for p in target_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        if stem.isdigit():
            try:
                maxnum = max(maxnum, int(stem))
            except Exception:
                pass
    return f"{maxnum + 1:0{pad}d}.jpg"


def commit_coco(staged_json: Dict[str, Any], staging_dir: Path, dataset_dir: Path, split: str, mode: str, rename: bool, pad: int, dry_run: bool, metadata_map: Dict[str, int] | None = None) -> None:
    target_path, target = ensure_target_coco(dataset_dir, split, staged_json)

    img_id_next, ann_id_next = next_ids(target)

    # build category mapping (optionally aligned to metadata_map)
    cat_map = build_cat_map(target, staged_json, metadata_map)

    target_imgs_by_name = {img.get("file_name"): img for img in target.get("images", [])}

    target_dir = dataset_dir / "coco" / split
    target_dir.mkdir(parents=True, exist_ok=True)

    planned_moves: List[str] = []

    for simg in staged_json.get("images", []) or []:
        fname = simg.get("file_name")
        src_img_path = staging_dir / fname
        if not src_img_path.exists():
            print(f"Warning: staged image file not found: {src_img_path}; skipping")
            continue

        # decide new filename
        if rename:
            new_fname = f"{img_id_next:0{pad}d}.jpg"
        else:
            # avoid collisions
            if fname in target_imgs_by_name:
                # pick new name based on next number
                new_fname = compute_next_filename(target_dir, pad)
            else:
                new_fname = fname

        new_img_path = target_dir / new_fname

        action = f"{mode} {src_img_path} -> {new_img_path}"
        planned_moves.append(action)

        if not dry_run:
            if mode == "move":
                shutil.move(str(src_img_path), str(new_img_path))
            else:
                shutil.copy2(str(src_img_path), str(new_img_path))

        # register image entry with new id
        new_img_entry = dict(simg)
        new_img_entry["id"] = img_id_next
        new_img_entry["file_name"] = new_fname
        target.setdefault("images", []).append(new_img_entry)

        # map annotations for this image
        for ann in staged_json.get("annotations", []) or []:
            if int(ann.get("image_id", -1)) != int(simg.get("id", -1)):
                continue
            new_ann = dict(ann)
            new_ann["id"] = ann_id_next
            new_ann["image_id"] = img_id_next
            # remap category id
            old_cid = int(ann.get("category_id", 0))
            new_ann["category_id"] = cat_map.get(old_cid, old_cid)
            target.setdefault("annotations", []).append(new_ann)
            ann_id_next += 1

        img_id_next += 1

    # write target json (backup first)
    if dry_run:
        print("DRY RUN - planned actions:")
        for m in planned_moves:
            print("  ", m)
        print(f"Would update COCO JSON at: {target_path}")
        return

    # backup
    if target_path.exists():
        bak = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(str(target_path), str(bak))
        print(f"Backed up existing COCO JSON to: {bak}")

    write_json(target_path, target)
    print(f"Wrote updated COCO JSON to: {target_path}")


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}", file=sys.stderr)
        return 2

    staging_dir = dataset_dir / args.staging_subdir
    if not staging_dir.exists():
        print(f"Staging folder not found: {staging_dir}", file=sys.stderr)
        return 3

    jsons = find_coco_jsons(staging_dir)
    if not jsons:
        print(f"No JSON files found in staging: {staging_dir}", file=sys.stderr)
        return 4

    # for now process each JSON found
    # load metadata mapping if requested
    metadata_map = None
    if args.align_metadata:
        md_path = dataset_dir / "METADATA.json"
        if md_path.exists():
            try:
                md = load_json(md_path)
                class_names = md.get("class_names", []) or []
                metadata_map = {name: idx for idx, name in enumerate(class_names)}
                print(f"Aligning staged categories to METADATA.json class_names: {class_names}")
            except Exception as e:
                print(f"Failed to read METADATA.json for alignment: {e}")
                metadata_map = None
        else:
            print(f"METADATA.json not found at {md_path}; --align-metadata ignored")

    for j in jsons:
        try:
            staged = load_json(j)
        except Exception as e:
            print(f"Failed to load JSON {j}: {e}", file=sys.stderr)
            continue

        print(f"Processing staged JSON: {j}")
        # If split distribution provided, split staged images across requested splits
        if args.split_distribution:
            # parse distribution string
            parts = [p.strip() for p in args.split_distribution.split(",") if p.strip()]
            dist: Dict[str, float] = {}
            for part in parts:
                if ":" not in part:
                    print(f"Invalid split distribution entry: {part}")
                    continue
                sname, sval = part.split(":", 1)
                try:
                    dist[sname.strip()] = float(sval)
                except Exception:
                    print(f"Invalid percentage for split '{sname}': {sval}")

            # normalize percentages
            total = sum(dist.values())
            if total <= 0:
                print("Split distribution sums to zero; skipping distribution.")
                continue
            for k in list(dist.keys()):
                dist[k] = dist[k] / total

            imgs = staged.get("images", []) or []
            n = len(imgs)
            if n == 0:
                print("No images in staged JSON; nothing to distribute.")
                continue

            # determine counts using largest-remainder method
            raw_counts: Dict[str, float] = {k: dist[k] * n for k in dist}
            floor_counts: Dict[str, int] = {k: int(raw_counts[k]) for k in raw_counts}
            remainder = n - sum(floor_counts.values())
            # sort by fractional part descending
            frac_sorted = sorted(raw_counts.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)
            counts: Dict[str, int] = dict(floor_counts)
            idx = 0
            while remainder > 0 and idx < len(frac_sorted):
                k = frac_sorted[idx][0]
                counts[k] += 1
                remainder -= 1
                idx += 1

            # build assignment order
            if args.assign_mode == "sequential":
                imgs_sorted = sorted(imgs, key=lambda x: x.get("file_name", ""))
            else:
                import random

                rnd = random.Random(args.seed)
                imgs_sorted = imgs[:]
                rnd.shuffle(imgs_sorted)

            # slice imgs_sorted into per-split lists
            pos = 0
            for sname, cnt in counts.items():
                part_imgs = imgs_sorted[pos : pos + cnt]
                pos += cnt
                if not part_imgs:
                    continue
                img_ids = {int(i.get("id")): True for i in part_imgs}
                part_anns = [a for a in (staged.get("annotations") or []) if int(a.get("image_id", -1)) in img_ids]
                part_json = {
                    "info": staged.get("info", {}),
                    "images": part_imgs,
                    "annotations": part_anns,
                    "categories": staged.get("categories", []),
                }
                print(f"Assigning {len(part_imgs)} images to split '{sname}'")
                commit_coco(part_json, staging_dir, dataset_dir, sname, args.mode, args.rename, args.pad, args.dry_run, metadata_map)
        else:
            commit_coco(staged, staging_dir, dataset_dir, args.split, args.mode, args.rename, args.pad, args.dry_run, metadata_map)

        if not args.dry_run:
            # after successful merge, remove the staged JSON
            try:
                j.unlink()
                print(f"Removed staged JSON: {j}")
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
