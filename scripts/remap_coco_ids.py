#!/usr/bin/env python3
"""Remap COCO annotation category_id values to 1-based indices.

Creates a `.bak` backup for each modified JSON.

Usage:
  python scripts/remap_coco_ids.py --dataset-dir datasets/<UUID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", "-d", required=True)
    return p.parse_args()


def remap(path: Path) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    # backup
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_text(json.dumps(data, indent=2), encoding="utf-8")

    anns = data.get("annotations", [])
    for a in anns:
        try:
            a["category_id"] = int(a.get("category_id", 0)) + 1
        except Exception:
            pass

    cats = data.get("categories", None)
    if isinstance(cats, list) and cats:
        for i, c in enumerate(cats, start=1):
            try:
                c["id"] = i
            except Exception:
                pass

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def main():
    args = parse_args()
    ds = Path(args.dataset_dir)
    for split in ("train", "valid", "test"):
        p = ds / "coco" / split / "_annotations.coco.json"
        ok = remap(p)
        if ok:
            print(f"Patched: {p}")
        else:
            print(f"Skipped (not found): {p}")


if __name__ == "__main__":
    main()
