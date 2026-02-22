#!/usr/bin/env python3
"""Create a dataset folder following the project's UUID layout.

Usage examples:
  python scripts/create_dataset.py
  python scripts/create_dataset.py --uuid 123e4567-e89b-12d3-a456-426614174000
  python scripts/create_dataset.py --root C:\\data --name "my-dataset" --force

This script creates the following structure under `<root>/<UUID>/`:
  raw/
  coco/train/ coco/valid/ coco/test/
  yolo/
  models/
  logs/
  exports/
  METADATA.json (basic manifest)
  README.md (brief guidance)
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create dataset UUID folder structure")
    p.add_argument("--uuid", "-u", help="UUID to use for the dataset (default: generated)")
    p.add_argument("--root", "-r", default="datasets", help="Root directory to create the dataset under (default: ./datasets)")
    p.add_argument("--name", "-n", default=None, help="Optional short name/label for the dataset")
    p.add_argument("--force", "-f", action="store_true", help="If set, will not error when dataset dir exists (will ensure subfolders exist)")
    p.add_argument("--no-readme", action="store_true", help="Do not create README.md")
    p.add_argument("--classes", "-c", action="append", help=(
        "Class names for the dataset. Accepts comma-separated or space-separated names. "
        "Can be supplied multiple times, e.g. -c cat,dog -c bird or -c cat dog bird"))
    p.add_argument("--classes-file", help="Path to a file listing class names (one per line)")
    return p.parse_args()


def ensure_uuid(u: str) -> str:
    try:
        return str(uuid.UUID(u))
    except Exception:
        raise ValueError(f"Invalid UUID: {u}")


def create_structure(root: Path, uuid_str: str, name: str | None, force: bool, no_readme: bool, class_names: list[str] | None = None) -> Path:
    dataset_dir = (root / uuid_str).resolve()

    if dataset_dir.exists() and not force:
        raise FileExistsError(f"Dataset directory already exists: {dataset_dir} (use --force to proceed)")

    # recommended subfolders
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
        p = dataset_dir / sp
        p.mkdir(parents=True, exist_ok=True)

    # METADATA.json
    metadata_path = dataset_dir / "METADATA.json"
    metadata = {
        "uuid": uuid_str,
        "name": name or "",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "class_names": class_names or [],
        "notes": "",
    }
    if not metadata_path.exists() or force:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # README.md
    if not no_readme:
        readme_path = dataset_dir / "README.md"
        if not readme_path.exists() or force:
            readme_text = f"# Dataset {uuid_str}\n\n"
            if name:
                readme_text += f"Name: {name}\n\n"
            readme_text += (
                "This folder follows the project CONVENTION for RF-DETR training.\n"
                "See `docs/CONTEXT` in the project root for details about layout and training.\n\n"
                "COCO-format data should be placed under `coco/train`, `coco/valid`, and `coco/test`.\n\n"
                "A `staging/` area is provided to support external labeling workflows:\n"
                "- staging/to_label: drop images here for labeling with external tools\n"
                "- staging/labeled: move labeled images (and label files) here when ready to commit\n"
                "- staging/quarantine: problematic files for inspection\n\n"
                "Use a commit script (e.g. `scripts/commit_from_staging.py`) to atomically move and rename\n"
                "staged images into `raw/` or `coco/<split>/` and to merge label files into COCO JSONs.\n"
            )
            readme_path.write_text(readme_text, encoding="utf-8")

    return dataset_dir


def main() -> int:
    args = parse_args()

    if args.uuid:
        try:
            uuid_str = ensure_uuid(args.uuid)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
    else:
        uuid_str = str(uuid.uuid4())

    # build class list from CLI args and/or file
    cls: list[str] = []
    if args.classes:
        for entry in args.classes:
            # split by comma and whitespace
            parts = [p.strip() for p in entry.replace(",", " ").split() if p.strip()]
            cls.extend(parts)

    if args.classes_file:
        cf = Path(args.classes_file).expanduser().resolve()
        if not cf.exists():
            print(f"Classes file not found: {cf}", file=sys.stderr)
            return 5
        for line in cf.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                cls.append(name)

    # preserve order but deduplicate
    seen = set()
    classes_final = []
    for c in cls:
        if c not in seen:
            seen.add(c)
            classes_final.append(c)

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    try:
        dataset_dir = create_structure(root, uuid_str, args.name, args.force, args.no_readme, classes_final)
    except FileExistsError as e:
        print(e, file=sys.stderr)
        return 3
    except Exception as e:
        print(f"Failed to create dataset: {e}", file=sys.stderr)
        return 4

    print(f"Created dataset structure at: {dataset_dir}")
    print("Subfolders: raw, coco/train, coco/valid, coco/test, yolo, models, logs, exports")
    if classes_final:
        print(f"METADATA.json written with class_names: {classes_final} and README.md added unless --no-readme used")
    else:
        print("METADATA.json written (class_names empty) and README.md added unless --no-readme used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
