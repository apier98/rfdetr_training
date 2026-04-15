"""GBT training for the startup-suggestion offline prior (Tier 1).

Trains one LightGBM model per prediction target using training rows produced
by MoldTrace.  Models are exported to ONNX via ``onnxmltools`` so that
MoldPilot can run inference with its existing ONNX Runtime installation —
no Python ML dependencies on the shop floor.

Targets
-------
- ``quality_score``     — regression  (``y_quality_score``, float 0–1)
- ``defect_burn_mark``  — binary classification (``y_defect_burn_mark``, 0/1)
- ``defect_flash``      — binary classification (``y_defect_flash``, 0/1)
- ``defect_sink_mark``  — binary classification (``y_defect_sink_mark``, 0/1)
- ``defect_weld_line``  — binary classification (``y_defect_weld_line``, 0/1)

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
    extract_parameter_schema,
    extract_targets,
    filter_eligible,
    infer_feature_keys,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUGGESTION_TARGETS: Dict[str, Tuple[Literal["regression", "classification"], str]] = {
    "quality_score":    ("regression",     "y_quality_score"),
    "defect_burn_mark": ("classification", "y_defect_burn_mark"),
    "defect_flash":     ("classification", "y_defect_flash"),
    "defect_sink_mark": ("classification", "y_defect_sink_mark"),
    "defect_weld_line": ("classification", "y_defect_weld_line"),
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
    #: Minimum eligible rows required to attempt training.
    min_rows: int = 10


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------

@dataclass
class TargetResult:
    target_name: str
    model_type: Literal["regression", "classification"]
    model: Any                      # fitted LightGBM model
    feature_keys: List[str]
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
        from sklearn.metrics import mean_squared_error, roc_auc_score
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
    raw_matrix, _ = align_to_union_schema(eligible, union_keys=union_feature_keys)
    raw_matrix, feature_keys, feature_presence = _prune_sparse_features(
        raw_matrix,
        union_feature_keys,
        min_presence_ratio=config.min_feature_presence_ratio,
    )
    parameter_schema = extract_parameter_schema(eligible, feature_keys=feature_keys)

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

    for target_name, (model_type, jsonl_key) in SUGGESTION_TARGETS.items():
        raw_y = extract_targets(eligible, jsonl_key)

        # Filter rows where target is present.
        valid_mask = [v is not None for v in raw_y]
        X_valid = X[valid_mask]
        y_valid = np.array([v for v in raw_y if v is not None], dtype=np.float32)

        if len(y_valid) < config.min_rows:
            # Not enough data for this target — skip silently with NaN metric.
            continue

        n_total = len(y_valid)
        split_idx = max(1, int(n_total * 0.8))
        X_train, X_eval = X_valid[:split_idx], X_valid[split_idx:]
        y_train, y_eval = y_valid[:split_idx], y_valid[split_idx:]

        if model_type == "regression":
            model = lgb.LGBMRegressor(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                num_leaves=config.num_leaves,
                min_child_samples=config.min_child_samples,
                n_jobs=config.n_jobs,
                random_state=config.random_state,
            )
            cv = KFold(n_splits=min(config.cv_folds, len(y_train)), shuffle=True, random_state=config.random_state)
            cv_scores = _cv_score_regression(model, X_train, y_train, cv, lgb, config)
            model.fit(
                X_train, y_train,
                eval_set=[(X_eval, y_eval)] if len(X_eval) > 0 else None,
                callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False), lgb.log_evaluation(-1)]
                if len(X_eval) > 0 else [lgb.log_evaluation(-1)],
            )
            metric_name = "rmse"
            metric_value = float(np.mean(cv_scores))
            metric_std = float(np.std(cv_scores))

        else:  # classification
            # Guard: need both classes present.
            if len(np.unique(y_train)) < 2:
                continue
            model = lgb.LGBMClassifier(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                num_leaves=config.num_leaves,
                min_child_samples=config.min_child_samples,
                n_jobs=config.n_jobs,
                random_state=config.random_state,
                class_weight="balanced",
            )
            cv = StratifiedKFold(n_splits=min(config.cv_folds, len(y_train)), shuffle=True, random_state=config.random_state)
            cv_scores = _cv_score_classification(model, X_train, y_train, cv, lgb, config)
            model.fit(
                X_train, y_train,
                eval_set=[(X_eval, y_eval)] if len(X_eval) > 0 else None,
                callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False), lgb.log_evaluation(-1)]
                if len(X_eval) > 0 else [lgb.log_evaluation(-1)],
            )
            metric_name = "auc"
            metric_value = float(np.mean(cv_scores))
            metric_std = float(np.std(cv_scores))

        target_results[target_name] = TargetResult(
            target_name=target_name,
            model_type=model_type,
            model=model,
            feature_keys=feature_keys,
            n_train=len(X_train),
            n_eval=len(X_eval),
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
        targets=target_results,
        null_strategy=config.null_strategy,
        n_eligible_rows=len(eligible),
        config=config,
    )


# ---------------------------------------------------------------------------
# CV helpers (avoids direct sklearn pipeline to keep LightGBM callbacks)
# ---------------------------------------------------------------------------

def _cv_score_regression(model, X, y, cv, lgb, config) -> List[float]:
    import numpy as np
    from sklearn.metrics import mean_squared_error

    scores = []
    for train_idx, val_idx in cv.split(X):
        m = model.__class__(**model.get_params())
        m.fit(X[train_idx], y[train_idx], callbacks=[lgb.log_evaluation(-1)])
        preds = m.predict(X[val_idx])
        scores.append(math.sqrt(mean_squared_error(y[val_idx], preds)))
    return scores


def _cv_score_classification(model, X, y, cv, lgb, config) -> List[float]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
            continue
        m = model.__class__(**model.get_params())
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
