#!/usr/bin/env python3
"""DEPRECATED wrapper for training.

Use instead:
  python -m rfdetr_training train ...
"""

from __future__ import annotations

import sys

from rfdetr_training.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))

