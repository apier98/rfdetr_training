"""Load, validate and extract feature matrices from training row JSONL.

Supports both ``training_row_v1`` (legacy spread statistics) and
``training_row_v2`` (setpoint-centric features with active_ratio).

This module is the bridge between MoldTrace (which produces supervised rows)
and MoldVision predictive training (which consumes them).  It is intentionally
dependency-free (no numpy/pandas) so that validation can run even in minimal
environments.  Downstream training code converts the nested-list output to
numpy/pandas as needed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# -- schema constants --------------------------------------------------------

SCHEMA_VERSION = "training_row_v2"
ACCEPTED_SCHEMA_VERSIONS = frozenset(["training_row_v1", "training_row_v2"])

REQUIRED_TOP_KEYS = frozenset(
    ["schema_version", "session_id", "component_id", "eligibility",
     "features", "targets", "context"]
)

REQUIRED_ELIGIBILITY_KEYS = frozenset(
    ["training_ready", "base_quality_gate_ready", "coverage_ratio"]
)

REQUIRED_TARGET_KEYS = frozenset(["y_quality_score"])

REQUIRED_CONTEXT_KEYS = frozenset(
    ["defect_classes_monitored", "feature_keys"]
)


# -- loading -----------------------------------------------------------------

def load_training_rows(path: Path) -> List[dict]:
    """Load ``training_row_v1`` records from a JSONL file.

    Blank lines are silently skipped.  Malformed JSON lines raise on the
    first bad line with a message that includes the line number.
    """
    path = Path(path)
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {lineno} of {path}: {exc}"
                ) from exc
            rows.append(obj)
    return rows


# -- validation --------------------------------------------------------------

def validate_row(row: dict) -> List[str]:
    """Validate a single ``training_row_v1`` record.

    Returns a (possibly empty) list of human-readable error strings.
    A row with zero errors is considered schema-compliant.
    """
    errors: List[str] = []

    if not isinstance(row, dict):
        return [f"Row is {type(row).__name__}, expected dict"]

    sv = row.get("schema_version")
    if sv not in ACCEPTED_SCHEMA_VERSIONS:
        errors.append(
            f"schema_version is {sv!r}, expected one of {sorted(ACCEPTED_SCHEMA_VERSIONS)}"
        )

    missing_top = REQUIRED_TOP_KEYS - set(row.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {sorted(missing_top)}")

    elig = row.get("eligibility")
    if isinstance(elig, dict):
        missing_elig = REQUIRED_ELIGIBILITY_KEYS - set(elig.keys())
        if missing_elig:
            errors.append(f"Missing eligibility keys: {sorted(missing_elig)}")
    elif elig is not None:
        errors.append("eligibility is not a dict")

    targets = row.get("targets")
    if isinstance(targets, dict):
        missing_tgt = REQUIRED_TARGET_KEYS - set(targets.keys())
        if missing_tgt:
            errors.append(f"Missing target keys: {sorted(missing_tgt)}")
    elif targets is not None:
        errors.append("targets is not a dict")

    ctx = row.get("context")
    if isinstance(ctx, dict):
        missing_ctx = REQUIRED_CONTEXT_KEYS - set(ctx.keys())
        if missing_ctx:
            errors.append(f"Missing context keys: {sorted(missing_ctx)}")
    elif ctx is not None:
        errors.append("context is not a dict")

    features = row.get("features")
    if isinstance(features, dict):
        if len(features) == 0:
            errors.append("features dict is empty")
    elif features is not None:
        errors.append("features is not a dict")

    return errors


def validate_dataset(rows: Sequence[dict]) -> Dict[str, Any]:
    """Validate a full dataset of ``training_row_v1`` records.

    Returns a summary dict with per-row errors, overall counts, and a
    ``valid`` flag (True only when every row passes).
    """
    row_errors: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        errs = validate_row(row)
        if errs:
            row_errors.append({
                "index": idx,
                "session_id": row.get("session_id"),
                "component_id": row.get("component_id"),
                "errors": errs,
            })
    return {
        "total_rows": len(rows),
        "valid_rows": len(rows) - len(row_errors),
        "invalid_rows": len(row_errors),
        "valid": len(row_errors) == 0,
        "row_errors": row_errors,
    }


# -- filtering ---------------------------------------------------------------

def filter_eligible(rows: Sequence[dict]) -> List[dict]:
    """Return only rows where ``eligibility.training_ready`` is truthy."""
    return [
        r for r in rows
        if (r.get("eligibility") or {}).get("training_ready")
    ]


# -- feature extraction ------------------------------------------------------

def infer_feature_keys(rows: Sequence[dict]) -> List[str]:
    """Derive the union of feature column names across all rows.

    Uses ``context.feature_keys`` when available, falling back to the actual
    keys in the ``features`` dict.  Returns a sorted deterministic list.
    """
    keys: set[str] = set()
    for row in rows:
        ctx = row.get("context") or {}
        fk = ctx.get("feature_keys")
        if isinstance(fk, list) and fk:
            keys.update(fk)
        else:
            feats = row.get("features")
            if isinstance(feats, dict):
                keys.update(feats.keys())
    return sorted(keys)


def check_schema_homogeneity(rows: Sequence[dict]) -> Dict[str, Any]:
    """Check whether all rows share the same feature schema.

    Returns a dict with ``homogeneous`` (bool), ``n_schemas``, and per-layout
    details.  Mirrors the ``schema_homogeneous`` logic in MoldTrace's
    ``export_training_rows`` summary.
    """
    seen_sets: dict[frozenset[str], int] = {}
    layouts: List[dict] = []
    for row in rows:
        ctx = row.get("context") or {}
        fk = ctx.get("feature_keys")
        if isinstance(fk, list):
            fs = frozenset(fk)
        else:
            fs = frozenset((row.get("features") or {}).keys())
        seen_sets[fs] = seen_sets.get(fs, 0) + 1
        layout_entry = {
            "hmi_layout_id": ctx.get("hmi_layout_id"),
            "machine_family": ctx.get("machine_family"),
        }
        if layout_entry not in layouts:
            layouts.append(layout_entry)
    return {
        "homogeneous": len(seen_sets) <= 1,
        "n_schemas": len(seen_sets),
        "rows_per_schema": dict(
            (str(sorted(fs))[:80], cnt) for fs, cnt in seen_sets.items()
        ),
        "hmi_layouts_seen": layouts,
    }


def extract_feature_matrix(
    rows: Sequence[dict],
    feature_keys: Optional[List[str]] = None,
) -> Tuple[List[List[Optional[float]]], List[str]]:
    """Extract a feature matrix from training rows.

    Returns ``(matrix, column_names)`` where *matrix* is a list of rows and
    each inner list has one entry per *feature_key*.  Missing features are
    ``None`` (the downstream model is expected to handle NaN / imputation).

    If *feature_keys* is ``None``, the union of all row feature keys is used
    (via ``infer_feature_keys``).
    """
    if feature_keys is None:
        feature_keys = infer_feature_keys(rows)
    matrix: List[List[Optional[float]]] = []
    for row in rows:
        feats = row.get("features") or {}
        vector: List[Optional[float]] = []
        for key in feature_keys:
            val = feats.get(key)
            if val is None:
                vector.append(None)
            else:
                try:
                    f = float(val)
                    vector.append(None if math.isnan(f) else f)
                except (TypeError, ValueError):
                    vector.append(None)
        matrix.append(vector)
    return matrix, feature_keys


def extract_targets(
    rows: Sequence[dict],
    target_key: str = "y_quality_score",
) -> List[Optional[float]]:
    """Extract a single target column from training rows."""
    out: List[Optional[float]] = []
    for row in rows:
        targets = row.get("targets") or {}
        val = targets.get(target_key)
        if val is None:
            out.append(None)
        else:
            try:
                out.append(float(val))
            except (TypeError, ValueError):
                out.append(None)
    return out


# -- dataset summary ---------------------------------------------------------

def summarize_dataset(rows: Sequence[dict]) -> Dict[str, Any]:
    """Produce a human-readable summary of a training row dataset (v1 or v2)."""
    eligible = filter_eligible(rows)
    homogeneity = check_schema_homogeneity(rows)
    feature_keys = infer_feature_keys(rows)
    quality_scores = [
        t for t in extract_targets(eligible, "y_quality_score")
        if t is not None
    ]
    defect_counts: Dict[str, int] = {}
    for row in eligible:
        targets = row.get("targets") or {}
        for key, val in targets.items():
            if key.startswith("y_defect_") and val == 1:
                label = key[len("y_defect_"):]
                defect_counts[label] = defect_counts.get(label, 0) + 1

    return {
        "total_rows": len(rows),
        "eligible_rows": len(eligible),
        "feature_columns": len(feature_keys),
        "schema_homogeneous": homogeneity["homogeneous"],
        "n_schemas": homogeneity["n_schemas"],
        "hmi_layouts_seen": homogeneity["hmi_layouts_seen"],
        "quality_score_stats": {
            "count": len(quality_scores),
            "min": round(min(quality_scores), 4) if quality_scores else None,
            "max": round(max(quality_scores), 4) if quality_scores else None,
            "mean": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else None,
        },
        "defect_counts": dict(sorted(defect_counts.items())),
    }


# -- union-schema alignment (B1) -------------------------------------------

def compute_union_feature_keys(rows: Sequence[dict]) -> List[str]:
    """Return sorted union of all ``context.feature_keys`` across rows.

    This is the superset schema needed to train on pooled data from
    heterogeneous HMI layouts.
    """
    return infer_feature_keys(rows)


def align_to_union_schema(
    rows: Sequence[dict],
    union_keys: Optional[List[str]] = None,
) -> Tuple[List[List[Optional[float]]], List[str]]:
    """Align rows from heterogeneous layouts to a shared union schema.

    Missing columns are filled with ``None`` (GBT models handle NaN natively).
    This is a convenience wrapper around ``extract_feature_matrix``.
    """
    if union_keys is None:
        union_keys = compute_union_feature_keys(rows)
    return extract_feature_matrix(rows, feature_keys=union_keys)
