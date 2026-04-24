"""Write and pack startup-suggestion bundles (.sugbundle).

A ``.sugbundle`` is a renamed ``.zip`` file containing:

- ``manifest.json``             — identity, feature schema, target model metadata, checksums
- ``model_quality_score.onnx``  — LightGBM regression → float quality score
- ``model_defect_*.onnx``       — LightGBM regression → continuous per-defect burden signals
- ``training_meta.json``        — provenance (n_rows, CV metrics, date, dataset source)

The bundle is consumed by MoldPilot's ``SuggestionBundleReader`` which uses the
existing ONNX Runtime installation (no Python ML deps required at runtime).

Usage::

    from moldvision.predictive.suggestion_bundle import write_suggestion_bundle, pack_sugbundle
    bundle_dir = write_suggestion_bundle(
        output_dir=Path("datasets/UUID/predictive"),
        train_result=result,
        model_name="startup-suggestion",
        model_version="1.0.0",
    )
    sugbundle_path = pack_sugbundle(bundle_dir)
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .trainer import TrainResult, export_target_to_onnx

# Default quality weights (must match §8.3.1 of ARIA_System_Integration.md).
DEFAULT_QUALITY_WEIGHTS: Dict[str, float] = {
    "burn_mark": 0.20,
    "flash":     0.30,
    "sink_mark": 0.35,
    "weld_line": 0.15,
}

BUNDLE_SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_checksums(bundle_dir: Path, exclude: Optional[List[str]] = None) -> Dict[str, str]:
    exclude = exclude or []
    return {
        p.name: _sha256_file(p)
        for p in sorted(bundle_dir.iterdir())
        if p.is_file() and p.name not in exclude
    }


# ---------------------------------------------------------------------------
# Write bundle
# ---------------------------------------------------------------------------

def write_suggestion_bundle(
    output_dir: Path,
    train_result: TrainResult,
    *,
    model_name: str,
    model_version: str,
    channel: str = "stable",
    supersedes: Optional[str] = None,
    min_app_version: str = "0.0.0",
    quality_weights: Optional[Dict[str, float]] = None,
    dataset_source: Optional[str] = None,
    mold_id: Optional[str] = None,
    material_id: Optional[str] = None,
    machine_id: Optional[str] = None,
) -> Path:
    """Write a startup-suggestion bundle directory.

    Parameters
    ----------
    output_dir:
        Parent directory.  A sub-directory named ``<bundle_id>/`` is created inside.
    train_result:
        Result from ``train_suggestion_models``.
    model_name:
        Human-readable model name (e.g. ``"startup-suggestion"``).
    model_version:
        Semantic version string (e.g. ``"1.0.0"``).
    channel:
        ``"stable"`` or ``"beta"``.
    supersedes:
        ``bundle_id`` of the bundle this replaces, or ``None``.
    min_app_version:
        Minimum MoldPilot version that can load this bundle.
    quality_weights:
        Per-defect weights for the quality score formula.
        Defaults to the canonical weights from §8.3.1.
    dataset_source:
        Optional path/UUID of the source dataset for traceability.
    mold_id:
        Mold identifier this bundle was trained for (scope axis 1).
    material_id:
        Material identifier this bundle was trained for (scope axis 2).
    machine_id:
        Machine identifier (or family) this bundle was trained for (scope axis 3).
        A model trained on data from a specific machine family will only produce
        reliable suggestions when run on the same family.

    Returns
    -------
    Path
        Path to the written bundle directory.
    """
    safe_name = model_name.lower().replace(" ", "-").replace("_", "-")
    bundle_id = f"{safe_name}-v{model_version}"
    bundle_dir = Path(output_dir) / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    weights = quality_weights or DEFAULT_QUALITY_WEIGHTS

    # Export each trained target to ONNX.
    target_models: Dict[str, dict] = {}
    for target_name, target_result in train_result.targets.items():
        onnx_filename = f"model_{target_name}.onnx"
        onnx_bytes = export_target_to_onnx(target_result)
        (bundle_dir / onnx_filename).write_bytes(onnx_bytes)
        target_models[target_name] = {
            "filename": onnx_filename,
            "model_type": target_result.model_type,
            "source_target": target_result.source_target,
            "signal_kind": target_result.signal_kind,
            "signal_role": target_result.signal_role,
        }

    # Write training metadata.
    cv_metrics = {
        name: {
            "model_type": r.model_type,
            "source_target": r.source_target,
            "signal_kind": r.signal_kind,
            "signal_role": r.signal_role,
            "metric": r.cv_metric_name,
            "mean": round(r.cv_metric_value, 6),
            "std": round(r.cv_metric_std, 6),
            "n_train": r.n_train,
            "n_eval": r.n_eval,
            "used_feature_keys": list(r.used_feature_keys),
        }
        for name, r in train_result.targets.items()
    }
    training_meta = {
        "n_eligible_rows": train_result.n_eligible_rows,
        "n_features": len(train_result.feature_keys),
        "selected_feature_keys": list(train_result.feature_keys),
        "trained_feature_keys": list(train_result.feature_keys),
        "null_strategy": train_result.null_strategy,
        "selected_feature_stats": list(train_result.config.selected_feature_stats),
        "control_family_validation": list(train_result.control_family_validation),
        "cv_metrics": cv_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": dataset_source,
        "lgbm_config": {
            "n_estimators": train_result.config.n_estimators,
            "learning_rate": train_result.config.learning_rate,
            "num_leaves": train_result.config.num_leaves,
            "min_feature_presence_ratio": train_result.config.min_feature_presence_ratio,
        },
    }
    _save_json(bundle_dir / "training_meta.json", training_meta)

    # Compute checksums before writing manifest (manifest itself excluded).
    checksums = _compute_checksums(bundle_dir, exclude=["manifest.json"])

    manifest = {
        "bundle_type":     "startup_suggestion",
        "bundle_id":       bundle_id,
        "model_name":      model_name,
        "model_version":   model_version,
        "schema_version":  BUNDLE_SCHEMA_VERSION,
        "channel":         channel,
        "supersedes":      supersedes,
        "min_app_version": min_app_version,
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "feature_keys":    train_result.feature_keys,
        "context_feature_keys": list(train_result.context_feature_keys),
        "trained_feature_keys": list(train_result.feature_keys),
        "parameter_schema": train_result.parameter_schema,
        "control_families": train_result.control_families,
        "deployable_control_families": train_result.deployable_control_families,
        "imputation_values": {
            k: round(v, 8) for k, v in train_result.imputation_values.items()
        },
        "null_strategy":   train_result.null_strategy,
        "selected_feature_stats": list(train_result.config.selected_feature_stats),
        "target_models":   target_models,
        "quality_weights": weights,
        "checksums":       checksums,
        "scope": {
            "mold_id":     mold_id,
            "material_id": material_id,
            "machine_id":  machine_id,
        },
    }
    _save_json(bundle_dir / "manifest.json", manifest)

    return bundle_dir


# ---------------------------------------------------------------------------
# Pack as .sugbundle
# ---------------------------------------------------------------------------

def pack_sugbundle(bundle_dir: Path) -> Path:
    """Zip *bundle_dir* into ``<bundle_id>.sugbundle`` alongside it.

    Parameters
    ----------
    bundle_dir:
        Directory produced by ``write_suggestion_bundle``.

    Returns
    -------
    Path
        Path to the created ``.sugbundle`` file.
    """
    archive_path = bundle_dir.with_suffix(".sugbundle")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(bundle_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(bundle_dir)))
    return archive_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
