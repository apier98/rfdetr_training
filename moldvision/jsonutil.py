"""Centralised JSON I/O helpers for the rfdetr_training package.

Use these instead of repeating ``json.loads(path.read_text(...))`` / inline
``try/except`` patterns throughout the codebase.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .pathutil import resolve_path

PathLike = Union[str, Path]


def load_json(path: PathLike, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read a JSON file and return its contents as a dict.

    Args:
        path:    Path to the JSON file (``str`` or :class:`~pathlib.Path`).
        default: Value to return when the file is missing or unreadable.
                 Defaults to an empty dict ``{}``.

    Returns:
        Parsed dict, or *default* on any error.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else (default if default is not None else {})
    except Exception:
        return default if default is not None else {}


def load_json_strict(path: PathLike) -> Dict[str, Any]:
    """Read a JSON file, raising on failure (no silent default).

    Use this when a missing or malformed file is always a programmer/user error.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: PathLike, obj: Dict[str, Any], *, indent: int = 2) -> None:
    """Write *obj* as indented JSON to *path*.

    Creates parent directories automatically.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=indent), encoding="utf-8")
