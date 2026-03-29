"""Centralised path-resolution helpers for the rfdetr_training package.

All code that needs to resolve a user-supplied path (str or Path, possibly
containing ``~``) should call :func:`resolve_path` instead of chaining
``.expanduser().resolve()`` inline. This makes the pattern easy to change in
one place and keeps call-sites clean.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def resolve_path(p: PathLike) -> Path:
    """Return an absolute, normalised :class:`~pathlib.Path`.

    Expands ``~`` / ``~user`` home-directory shorthands and resolves any
    ``..`` components.  Equivalent to ``Path(p).expanduser().resolve()`` but
    expressed as a single named call so it can be imported and reused
    throughout the package.
    """
    return Path(p).expanduser().resolve()
