#!/usr/bin/env python3
"""DEPRECATED wrapper for training.

Use instead:
  python -m moldvision train ...
"""

from __future__ import annotations

import sys

from moldvision.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))

