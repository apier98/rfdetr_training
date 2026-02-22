#!/usr/bin/env python3
"""Remove one or more category ids from COCO _annotations.coco.json files.

Backs up each JSON to a `.bak` file before modifying. Prints remaining category ids
found in annotations after removal.

Usage examples:
  python scripts/remove_coco_class.py --dataset-dir datasets/<UUID> --remove-id 0
  python scripts/remove_coco_class.py -d datasets/<UUID> --remove-ids 0,2
  python scripts/remove_coco_class.py -d datasets/<UUID> --remove-id 0 --remove-id 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Set, List


def parse_args():
    p = argparse.ArgumentParser(description="Remove category ids from COCO annotations")
    p.add_argument("--dataset-dir", "-d", required=True, help="Path to dataset UUID folder")
    p.add_argument("--remove-id", type=int, action="append", help="Category id to remove (repeatable)")
    p.add_argument("--remove-ids", help="Comma-separated list of ids to remove")
    p.add_argument("--remove-empty-images", action="store_true", help="Remove image entries that end up with zero annotations")
    return p.parse_args()


def _collect_remove_set(args) -> Set[int]:
    s: Set[int] = set()
    if args.remove_id:
        s.update(int(x) for x in args.remove_id)
    if args.remove_ids:
        for part in args.remove_ids.split(","):
            part = part.strip()
            if not part:
                continue
            s.add(int(part))
    return s


def process_file(path: Path, remove_set: Set[int], remove_empty_images: bool) -> List[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_text(json.dumps(data, indent=2), encoding="utf-8")

    anns = data.get("annotations", [])
    imgs = {im["id"]: im for im in data.get("images", [])}

    # filter annotations
    new_anns = [a for a in anns if int(a.get("category_id", 0)) not in remove_set]

    # optionally remove images that have no remaining anns
    remaining_image_ids = {a["image_id"] for a in new_anns}
    if remove_empty_images:
        new_images = [img for img_id, img in imgs.items() if img_id in remaining_image_ids]
    else:
        new_images = list(imgs.values())

    # rebuild categories: keep any category referenced by remaining annotations
    remaining_cat_ids = sorted({int(a["category_id"]) for a in new_anns})
    cats_map = {int(c.get("id", 0)): c for c in data.get("categories", [])}
    new_categories = [cats_map[cid] for cid in remaining_cat_ids if cid in cats_map]

    data["annotations"] = new_anns
    data["images"] = new_images
    data["categories"] = new_categories

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return remaining_cat_ids


def main():
    args = parse_args()
    remove_set = _collect_remove_set(args)
    if not remove_set:
        print("No category ids specified for removal. Use --remove-id or --remove-ids.")
        return 2

    ds = Path(args.dataset_dir)
    if not ds.exists():
        print(f"Dataset dir not found: {ds}")
        return 3

    all_remaining = set()
    for split in ("train", "valid", "test"):
        p = ds / "coco" / split / "_annotations.coco.json"
        if not p.exists():
            print(f"Skipped (not found): {p}")
            continue
        remaining = process_file(p, remove_set, args.remove_empty_images)
        print(f"Processed: {p}  Remaining category ids: {remaining}")
        all_remaining.update(remaining)

    print("Final remaining category ids across splits:", sorted(all_remaining))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
