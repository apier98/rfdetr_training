"""GBT training for the startup-suggestion offline prior (Tier 1).

Trains one LightGBM model per prediction target using training rows produced
by MoldTrace.  Models are exported to ONNX via ``onnxmltools`` so that
MoldPilot can run inference with its existing ONNX Runtime installation —
no Python ML dependencies on the shop floor.

Targets
-------
- ``quality_score``     — regression  (``y_quality_score``, float 0–1)
- ``defect_burn_mark``  — regression  (``y_burden_burn_mark``, float 0–1)
- ``defect_flash``      — regression  (``y_burden_flash``, float 0–1)
- ``defect_sink_mark``  — regression  (``y_burden_sink_mark``, float 0–1)
- ``defect_weld_line``  — regression  (``y_burden_weld_line``, float 0–1)

Usage::

    from moldvision.predictive.trainer import GbtTrainingConfig, train_suggestion_models
    rows = load_training_rows("training_rows.jsonl")
    result = train_suggestion_models(rows)
    print(result.cv_metrics)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .training_row_loader import (
    align_to_union_schema,
    extract_control_families,
    extract_parameter_schema,
    extract_targets,
    filter_eligible,
    infer_feature_keys,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUGGESTION_TARGETS: Dict[str, Tuple[Literal["regression", "classification"], str, str, str]] = {
    "quality_score":    ("regression", "y_quality_score", "quality_score", "optimization"),
    "defect_burn_mark": ("regression", "y_burden_burn_mark", "defect_burden", "optimization"),
    "defect_flash":     ("regression", "y_burden_flash", "defect_burden", "optimization"),
    "defect_sink_mark": ("regression", "y_burden_sink_mark", "defect_burden", "optimization"),
    "defect_weld_line": ("regression", "y_burden_weld_line", "defect_burden", "optimization"),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GbtTrainingConfig:
    """Hyper-parameters and training options for the GBT models."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 5
    n_jobs: int = -1
    random_state: int = 42
    cv_folds: int = 5
    early_stopping_rounds: int = 30
    null_strategy: Literal["native_missing", "mean_impute", "zero_impute"] = "native_missing"
    min_feature_presence_ratio: float = 0.05
    selected_feature_stats: Tuple[str, ...] = ("setpoint_end", "present")
    #: Minimum eligible rows required to attempt training.
    min_rows: int = 10


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------

@dataclass
class TargetResult:
    target_name: str
    model_type: Literal["regression", "classification"]
    source_target: str
    signal_kind: str
    signal_role: str
    model: Any                      # fitted LightGBM model
    feature_keys: List[str]
    used_feature_keys: List[str]
    n_train: int
    n_eval: int
    cv_metric_name: str             # "rmse" or "auc"
    cv_metric_value: float          # mean over folds
    cv_metric_std: float


@dataclass
class TrainResult:
    feature_keys: List[str]
    imputation_values: Dict[str, float]  # fallback values for legacy/runtime compatibility
    parameter_schema: List[Dict[str, Any]]
    control_families: List[Dict[str, Any]]
    deployable_control_families: List[Dict[str, Any]]
    control_family_validation: List[Dict[str, Any]]
    targets: Dict[str, TargetResult]
    null_strategy: Literal["native_missing", "mean_impute", "zero_impute"]
    n_eligible_rows: int
    config: GbtTrainingConfig = field(repr=False)


# ---------------------------------------------------------------------------
# Imputation helpers
# ---------------------------------------------------------------------------

def _compute_means(
    matrix: List[List[Optional[float]]],
    feature_keys: List[str],
) -> Dict[str, float]:
    """Compute per-column mean, ignoring None.  Fallback to 0.0 if all missing."""
    totals: Dict[str, float] = {k: 0.0 for k in feature_keys}
    counts: Dict[str, int] = {k: 0 for k in feature_keys}
    for row in matrix:
        for i, val in enumerate(row):
            if val is not None:
                totals[feature_keys[i]] += val
                counts[feature_keys[i]] += 1
    return {
        k: (totals[k] / counts[k] if counts[k] > 0 else 0.0)
        for k in feature_keys
    }


def _impute(
    matrix: List[List[Optional[float]]],
    fill_values: Dict[str, float],
    feature_keys: List[str],
) -> List[List[float]]:
    """Replace None with per-column fill value."""
    result: List[List[float]] = []
    for row in matrix:
        result.append([
            val if val is not None else fill_values.get(feature_keys[i], 0.0)
            for i, val in enumerate(row)
        ])
    return result


def _to_native_missing_matrix(
    matrix: List[List[Optional[float]]],
) -> List[List[float]]:
    """Replace None with NaN so LightGBM can learn native missing-value splits."""
    result: List[List[float]] = []
    for row in matrix:
        result.append([
            float(val) if val is not None else float("nan")
            for val in row
        ])
    return result


def _presence_ratio_by_feature(
    matrix: List[List[Optional[float]]],
    feature_keys: List[str],
) -> Dict[str, float]:
    n_rows = len(matrix)
    if n_rows <= 0:
        return {key: 0.0 for key in feature_keys}
    counts: Dict[str, int] = {key: 0 for key in feature_keys}
    for row in matrix:
        for idx, val in enumerate(row):
            if val is not None:
                counts[feature_keys[idx]] += 1
    return {
        key: (float(counts[key]) / float(n_rows))
        for key in feature_keys
    }


def _prune_sparse_features(
    matrix: List[List[Optional[float]]],
    feature_keys: List[str],
    *,
    min_presence_ratio: float,
) -> Tuple[List[List[Optional[float]]], List[str], Dict[str, float]]:
    ratios = _presence_ratio_by_feature(matrix, feature_keys)
    if not feature_keys:
        return matrix, feature_keys, ratios
    threshold = max(0.0, min(1.0, float(min_presence_ratio)))
    keep_indices = [
        idx for idx, key in enumerate(feature_keys)
        if ratios.get(key, 0.0) >= threshold
    ]
    if not keep_indices:
        keep_indices = list(range(len(feature_keys)))
    kept_keys = [feature_keys[idx] for idx in keep_indices]
    kept_matrix: List[List[Optional[float]]] = []
    for row in matrix:
        kept_matrix.append([row[idx] for idx in keep_indices])
    kept_ratios = {key: ratios[key] for key in kept_keys}
    return kept_matrix, kept_keys, kept_ratios


def _distinct_value_count_by_feature(
    matrix: List[List[Optional[float]]],
    feature_keys: List[str],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for idx, key in enumerate(feature_keys):
        values: set[float] = set()
        for row in matrix:
            val = row[idx]
            if val is None:
                continue
            values.add(float(val))
            if len(values) > 1:
                break
        counts[key] = len(values)
    return counts


def _prune_constant_features(
    matrix: List[List[Optional[float]]],
    feature_keys: List[str],
) -> Tuple[List[List[Optional[float]]], List[str], Dict[str, int]]:
    distinct_counts = _distinct_value_count_by_feature(matrix, feature_keys)
    keep_indices = [
        idx for idx, key in enumerate(feature_keys)
        if distinct_counts.get(key, 0) > 1
    ]
    kept_keys = [feature_keys[idx] for idx in keep_indices]
    kept_matrix: List[List[Optional[float]]] = []
    for row in matrix:
        kept_matrix.append([row[idx] for idx in keep_indices])
    kept_counts = {key: distinct_counts[key] for key in kept_keys}
    return kept_matrix, kept_keys, kept_counts


def _selected_feature_keys(
    feature_keys: Sequence[str],
    *,
    selected_stats: Sequence[str],
) -> List[str]:
    allowed_stats = {
        str(stat).strip()
        for stat in selected_stats
        if str(stat).strip()
    }
    if not allowed_stats:
        return list(feature_keys)

    filtered: List[str] = []
    for key in feature_keys:
        _, dot, stat = str(key).partition(".")
        if not dot:
            filtered.append(str(key))
            continue
        if stat in allowed_stats:
            filtered.append(str(key))

    # Preserve backward compatibility for legacy datasets that only contain
    # older stat names such as ".actual.mean".
    return filtered or list(feature_keys)


def _target_has_variation(values: Sequence[float]) -> bool:
    observed = {float(value) for value in values}
    return len(observed) > 1


def _annotate_trained_control_keys(
    parameter_schema: Sequence[Dict[str, Any]],
    trained_feature_keys: Sequence[str],
) -> List[Dict[str, Any]]:
    trained_set = {str(key) for key in trained_feature_keys}
    annotated: List[Dict[str, Any]] = []
    for item in parameter_schema:
        if not isinstance(item, dict):
            continue
        control_keys = [
            str(key)
            for key in item.get("control_feature_keys", ())
            if str(key).strip()
        ]
        annotated_item = dict(item)
        annotated_item["control_feature_keys"] = control_keys
        annotated_item["trained_control_feature_keys"] = [
            key for key in control_keys if key in trained_set
        ]
        annotated.append(annotated_item)
    return annotated


def _annotate_control_families(
    control_families: Sequence[Dict[str, Any]],
    trained_feature_keys: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    trained_set = {str(key) for key in trained_feature_keys}
    annotated: List[Dict[str, Any]] = []
    deployable: List[Dict[str, Any]] = []
    validation: List[Dict[str, Any]] = []

    for family in control_families:
        if not isinstance(family, dict):
            continue
        family_id = str(family.get("family_id") or "").strip()
        if not family_id:
            continue
        members_raw = family.get("ordered_members") or []
        if not isinstance(members_raw, list):
            members_raw = []
        family_type = str(family.get("family_type") or "").strip() or (
            "atomic" if len(members_raw) > 1 else "single_slot"
        )
        members: List[Dict[str, Any]] = []
        trained_member_parameter_ids: List[str] = []
        dropped_member_parameter_ids: List[str] = []

        for member in members_raw:
            if not isinstance(member, dict):
                continue
            parameter_id = str(member.get("parameter_id") or "").strip()
            if not parameter_id:
                continue
            control_feature_keys = [
                str(key)
                for key in member.get("control_feature_keys", ())
                if str(key).strip()
            ]
            trained_control_feature_keys = [
                key for key in control_feature_keys if key in trained_set
            ]
            if trained_control_feature_keys:
                trained_member_parameter_ids.append(parameter_id)
            elif control_feature_keys:
                dropped_member_parameter_ids.append(parameter_id)
            member_item = dict(member)
            member_item["control_feature_keys"] = control_feature_keys
            member_item["trained_control_feature_keys"] = trained_control_feature_keys
            members.append(member_item)

        effective_family_type = family_type
        deployable_members = list(members)
        if family_type == "atomic":
            partial_indices = [index for index, member in enumerate(members) if member.get("trained_control_feature_keys")]
            if bool(members) and not dropped_member_parameter_ids and bool(trained_member_parameter_ids):
                is_deployable = True
                reason = "ok"
            elif _is_contiguous_partial_subset(partial_indices, len(members)):
                effective_family_type = "partially_controllable"
                deployable_members = [members[index] for index in partial_indices]
                is_deployable = True
                reason = "partial_contiguous_subset"
            else:
                is_deployable = False
                reason = (
                    "atomic_member_missing_trained_feature"
                    if dropped_member_parameter_ids
                    else "no_trained_members"
                )
        else:
            deployable_members = [member for member in members if member.get("trained_control_feature_keys")]
            is_deployable = bool(trained_member_parameter_ids)
            reason = "ok" if is_deployable else "no_trained_members"

        family_constraints = dict(family.get("family_constraints") or {})
        if effective_family_type == "partially_controllable":
            family_constraints["controllable_member_parameter_ids"] = list(trained_member_parameter_ids)
            family_constraints["frozen_member_parameter_ids"] = list(dropped_member_parameter_ids)
            family_constraints["partial_family_policy"] = "contiguous_subset"

        family_item = dict(family)
        family_item["family_type"] = effective_family_type
        family_item["ordered_members"] = deployable_members
        family_item["all_ordered_members"] = members
        family_item["trained_member_parameter_ids"] = trained_member_parameter_ids
        family_item["dropped_member_parameter_ids"] = dropped_member_parameter_ids
        family_item["family_constraints"] = family_constraints
        family_item["deployable"] = is_deployable
        family_item["deployability_reason"] = reason
        annotated.append(family_item)
        validation.append(
            {
                "family_id": family_id,
                "family_type": effective_family_type,
                "deployable": is_deployable,
                "reason": reason,
                "trained_member_parameter_ids": list(trained_member_parameter_ids),
                "dropped_member_parameter_ids": list(dropped_member_parameter_ids),
            }
        )
        if is_deployable:
            deployable.append(family_item)

    return annotated, deployable, validation


def _is_contiguous_partial_subset(
    trained_indices: Sequence[int],
    total_members: int,
) -> bool:
    if total_members <= 1:
        return bool(trained_indices)
    if not trained_indices:
        return False
    ordered = sorted(set(int(index) for index in trained_indices))
    if len(ordered) <= 0:
        return False
    expected = list(range(ordered[0], ordered[-1] + 1))
    return ordered == expected


def _effective_min_child_samples(configured: int, n_rows: int) -> int:
    if n_rows <= 1:
        return 1
    return max(1, min(int(configured), n_rows // 2))


def _regressor_kwargs(config: GbtTrainingConfig, *, n_rows: int) -> Dict[str, Any]:
    return {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_child_samples": _effective_min_child_samples(config.min_child_samples, n_rows),
        "n_jobs": config.n_jobs,
        "random_state": config.random_state,
    }


def _classifier_kwargs(config: GbtTrainingConfig, *, n_rows: int) -> Dict[str, Any]:
    return {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_child_samples": _effective_min_child_samples(config.min_child_samples, n_rows),
        "n_jobs": config.n_jobs,
        "random_state": config.random_state,
        "class_weight": "balanced",
    }


def _effective_regression_cv_splits(n_rows: int, requested_splits: int) -> int:
    if n_rows < 2:
        return 0
    return min(int(requested_splits), int(n_rows))


def _effective_classification_cv_splits(y, requested_splits: int) -> int:
    import numpy as np

    if len(y) <= 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        return 0
    return max(0, min(int(requested_splits), int(len(y)), int(np.min(counts))))


def _used_feature_keys(model: Any, feature_keys: Sequence[str]) -> List[str]:
    import numpy as np

    raw_importances = getattr(model, "feature_importances_", None)
    if raw_importances is None:
        return list(feature_keys)
    importances = np.asarray(raw_importances).reshape(-1)
    if importances.size != len(feature_keys):
        return list(feature_keys)
    used = [
        str(feature_keys[idx])
        for idx, importance in enumerate(importances.tolist())
        if float(importance) > 0.0
    ]
    return used or list(feature_keys)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_suggestion_models(
    rows: Sequence[dict],
    config: Optional[GbtTrainingConfig] = None,
) -> TrainResult:
    """Train one LightGBM model per suggestion target.

    Parameters
    ----------
    rows:
        Training rows in ``training_row_v1`` or ``training_row_v2`` format,
        as returned by ``load_training_rows``.
    config:
        Training hyper-parameters.  Uses defaults if ``None``.

    Returns
    -------
    TrainResult
        Fitted models, feature schema, imputation values, and CV metrics.

    Raises
    ------
    ImportError
        When ``lightgbm`` is not installed.  Install with
        ``pip install aria-moldvision[predictive]``.
    ValueError
        When fewer than ``config.min_rows`` eligible rows are available.
    """
    try:
        import lightgbm as lgb
        import numpy as np
        from sklearn.model_selection import StratifiedKFold, KFold
    except ImportError as exc:
        raise ImportError(
            "lightgbm and scikit-learn are required for predictive training. "
            "Install with: pip install 'aria-moldvision[predictive]'"
        ) from exc

    if config is None:
        config = GbtTrainingConfig()

    eligible = filter_eligible(rows)
    if len(eligible) < config.min_rows:
        raise ValueError(
            f"Only {len(eligible)} eligible rows (training_ready=True). "
            f"Minimum required: {config.min_rows}. "
            "Accumulate more qualification sessions before training."
        )

    # Build feature matrix with union schema (None for missing columns), then
    # prune ultra-sparse features before training.
    union_feature_keys = infer_feature_keys(eligible)
    selected_feature_keys = _selected_feature_keys(
        union_feature_keys,
        selected_stats=config.selected_feature_stats,
    )
    raw_matrix, _ = align_to_union_schema(eligible, union_keys=selected_feature_keys)
    raw_matrix, feature_keys, _feature_presence = _prune_sparse_features(
        raw_matrix,
        selected_feature_keys,
        min_presence_ratio=config.min_feature_presence_ratio,
    )
    raw_matrix, feature_keys, _distinct_counts = _prune_constant_features(
        raw_matrix,
        feature_keys,
    )
    if not feature_keys:
        raise ValueError(
            "No informative feature columns remain after filtering. "
            "Eligible rows either keep the same parameter values throughout or "
            "controls appear only once. Accumulate more varied qualification data "
            "before training a scoped model."
        )
    parameter_schema = _annotate_trained_control_keys(
        extract_parameter_schema(eligible),
        feature_keys,
    )
    control_families, deployable_control_families, control_family_validation = _annotate_control_families(
        extract_control_families(eligible),
        feature_keys,
    )

    # Compute fallback values from training data for legacy/runtime compatibility.
    fill_values = _compute_means(raw_matrix, feature_keys)

    if config.null_strategy == "native_missing":
        X = np.array(_to_native_missing_matrix(raw_matrix), dtype=np.float32)
    elif config.null_strategy == "mean_impute":
        X = np.array(_impute(raw_matrix, fill_values, feature_keys), dtype=np.float32)
    else:
        zero_fill_values = {k: 0.0 for k in feature_keys}
        X = np.array(_impute(raw_matrix, zero_fill_values, feature_keys), dtype=np.float32)

    target_results: Dict[str, TargetResult] = {}

    for target_name, (model_type, jsonl_key, signal_kind, signal_role) in SUGGESTION_TARGETS.items():
        raw_y = extract_targets(eligible, jsonl_key)

        # Filter rows where target is present.
        valid_mask = [v is not None for v in raw_y]
        X_valid = X[valid_mask]
        y_valid = np.array([v for v in raw_y if v is not None], dtype=np.float32)

        if len(y_valid) < config.min_rows:
            # Not enough data for this target — skip silently with NaN metric.
            continue

        if not _target_has_variation(y_valid):
            # Constant targets do not add useful signal and only create
            # degenerate models / misleading perfect metrics in reports.
            continue

        if model_type == "regression":
            model_kwargs = _regressor_kwargs(config, n_rows=len(y_valid))
            model = lgb.LGBMRegressor(**model_kwargs)
            cv_splits = _effective_regression_cv_splits(len(y_valid), config.cv_folds)
            cv_scores = [0.0]
            if cv_splits >= 2:
                cv = KFold(n_splits=cv_splits, shuffle=True, random_state=config.random_state)
                cv_scores = _cv_score_regression(
                    lgb.LGBMRegressor,
                    model_kwargs,
                    X_valid,
                    y_valid,
                    cv,
                    lgb,
                )
            model.fit(
                X_valid,
                y_valid,
                callbacks=[lgb.log_evaluation(-1)],
            )
            used_feature_keys = _used_feature_keys(model, feature_keys)
            metric_name = "rmse"
            metric_value = float(np.mean(cv_scores))
            metric_std = float(np.std(cv_scores))

        else:  # classification
            # Guard: need both classes present.
            if len(np.unique(y_valid)) < 2:
                continue
            model_kwargs = _classifier_kwargs(config, n_rows=len(y_valid))
            model = lgb.LGBMClassifier(**model_kwargs)
            cv_splits = _effective_classification_cv_splits(y_valid, config.cv_folds)
            cv_scores = [0.5]
            if cv_splits >= 2:
                cv = StratifiedKFold(
                    n_splits=cv_splits,
                    shuffle=True,
                    random_state=config.random_state,
                )
                cv_scores = _cv_score_classification(
                    lgb.LGBMClassifier,
                    model_kwargs,
                    X_valid,
                    y_valid,
                    cv,
                    lgb,
                )
            model.fit(
                X_valid,
                y_valid,
                callbacks=[lgb.log_evaluation(-1)],
            )
            used_feature_keys = _used_feature_keys(model, feature_keys)
            metric_name = "auc"
            metric_value = float(np.mean(cv_scores))
            metric_std = float(np.std(cv_scores))

        target_results[target_name] = TargetResult(
            target_name=target_name,
            model_type=model_type,
            source_target=jsonl_key,
            signal_kind=signal_kind,
            signal_role=signal_role,
            model=model,
            feature_keys=feature_keys,
            used_feature_keys=used_feature_keys,
            n_train=len(X_valid),
            n_eval=0,
            cv_metric_name=metric_name,
            cv_metric_value=metric_value,
            cv_metric_std=metric_std,
        )

    if not target_results:
        raise ValueError(
            "No targets could be trained (insufficient data for all targets). "
            f"Check your training rows — only {len(eligible)} eligible rows available."
        )

    return TrainResult(
        feature_keys=feature_keys,
        imputation_values=fill_values,
        parameter_schema=parameter_schema,
        control_families=control_families,
        deployable_control_families=deployable_control_families,
        control_family_validation=control_family_validation,
        targets=target_results,
        null_strategy=config.null_strategy,
        n_eligible_rows=len(eligible),
        config=config,
    )


# ---------------------------------------------------------------------------
# CV helpers (avoids direct sklearn pipeline to keep LightGBM callbacks)
# ---------------------------------------------------------------------------

def _cv_score_regression(model_cls, model_kwargs, X, y, cv, lgb) -> List[float]:
    from sklearn.metrics import mean_squared_error

    scores = []
    for train_idx, val_idx in cv.split(X):
        m = model_cls(
            **{
                **model_kwargs,
                "min_child_samples": _effective_min_child_samples(
                    model_kwargs["min_child_samples"],
                    len(train_idx),
                ),
            }
        )
        m.fit(X[train_idx], y[train_idx], callbacks=[lgb.log_evaluation(-1)])
        preds = m.predict(X[val_idx])
        scores.append(math.sqrt(mean_squared_error(y[val_idx], preds)))
    return scores


def _cv_score_classification(model_cls, model_kwargs, X, y, cv, lgb) -> List[float]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
            continue
        m = model_cls(
            **{
                **model_kwargs,
                "min_child_samples": _effective_min_child_samples(
                    model_kwargs["min_child_samples"],
                    len(train_idx),
                ),
            }
        )
        m.fit(X[train_idx], y[train_idx], callbacks=[lgb.log_evaluation(-1)])
        proba = m.predict_proba(X[val_idx])[:, 1]
        scores.append(roc_auc_score(y[val_idx], proba))
    return scores if scores else [0.5]


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_target_to_onnx(target_result: TargetResult) -> bytes:
    """Export a single trained LightGBM model to ONNX bytes.

    Parameters
    ----------
    target_result:
        A ``TargetResult`` from ``train_suggestion_models``.

    Returns
    -------
    bytes
        Serialised ONNX model.

    Raises
    ------
    ImportError
        When ``onnxmltools`` or ``skl2onnx`` is not installed.
    """
    try:
        import onnxmltools
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as exc:
        raise ImportError(
            "onnxmltools is required for ONNX export. "
            "Install with: pip install 'aria-moldvision[predictive]'"
        ) from exc

    n_features = len(target_result.feature_keys)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(
        target_result.model,
        initial_types=initial_type,
        target_opset=15,
        zipmap=False,   # Return plain arrays, not ZipMap dicts
    )
    return onnx_model.SerializeToString()
