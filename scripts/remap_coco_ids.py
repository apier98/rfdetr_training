#!/usr/bin/env python3
"""DEPRECATED: this script remapped COCO category ids to 1-based indices.

RF-DETR training commonly expects contiguous 0-indexed ids (0..N-1). Use:
  python -m moldvision dataset normalize-coco-ids -d datasets/<UUID>

"""

from __future__ import annotations

import sys

from moldvision.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["dataset", "normalize-coco-ids", *sys.argv[1:]]))
