#!/usr/bin/env python3
"""DEPRECATED wrapper for dataset creation.

Use instead:
  python -m moldvision dataset create ...
"""

from __future__ import annotations

import sys

from moldvision.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["dataset", "create", *sys.argv[1:]]))

