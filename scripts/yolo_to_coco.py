#!/usr/bin/env python3
"""DEPRECATED wrapper for YOLO -> COCO conversion.

Use instead:
  python -m moldvision dataset yolo-to-coco ...
"""

from __future__ import annotations

import sys

from moldvision.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["dataset", "yolo-to-coco", *sys.argv[1:]]))

