"""Tests for the GBT startup-suggestion trainer.

These tests require the ``[predictive]`` optional deps (lightgbm, scikit-learn,
onnxmltools).  They are skipped automatically when the deps are not installed.
"""
from __future__ import annotations

import unittest
from typing import Optional

try:
    import lightgbm  # noqa: F401
    import numpy  # noqa: F401
    import sklearn  # noqa: F401
    _PREDICTIVE_AVAILABLE = True
except ImportError:
    _PREDICTIVE_AVAILABLE = False


def _make_row(
    *,
    session_id: str = "sess_001",
    component_id: str = "cmp_1",
    training_ready: bool = True,
    quality_score: Optional[float] = 0.80,
    defect_flash: int = 0,
    defect_sink_mark: int = 1,
    defect_burn_mark: int = 0,
    defect_weld_line: int = 0,
) -> dict:
    return {
        "schema_version": "training_row_v1",
        "session_id": session_id,
        "component_id": component_id,
        "eligibility": {
            "training_ready": training_ready,
            "base_quality_gate_ready": True,
            "coverage_ratio": 0.95,
        },
        "features": {
            "temp_barrel:actual.mean": 220.0 + (hash(session_id) % 20),
            "pressure_injection:actual.last": 850.0 + (hash(component_id) % 50),
            "injection_speed:actual.mean": 65.0,
        },
        "targets": {
            "y_quality_score": quality_score,
            "y_defect_flash": defect_flash,
            "y_defect_sink_mark": defect_sink_mark,
            "y_defect_burn_mark": defect_burn_mark,
            "y_defect_weld_line": defect_weld_line,
        },
        "context": {
            "defect_classes_monitored": ["flash", "sink_mark", "burn_mark", "weld_line"],
            "feature_keys": [
                "temp_barrel:actual.mean",
                "pressure_injection:actual.last",
                "injection_speed:actual.mean",
            ],
            "parameter_schema": [
                {
                    "parameter_id": "temp_barrel",
                    "display_name": "Barrel Temperature",
                    "unit": "C",
                    "baseline": 220.0,
                    "range_min": 180.0,
                    "range_max": 320.0,
                    "control_feature_keys": ["temp_barrel:actual.mean"],
                    "step_mode": "absolute",
                    "preferred_step": 1.0,
                    "max_delta": 5.0,
                }
            ],
        },
    }


def _make_rows(n: int = 30) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(_make_row(
            session_id=f"sess_{i:03d}",
            component_id=f"cmp_{i % 5}",
            quality_score=0.5 + (i % 10) * 0.04,
            defect_flash=int(i % 3 == 0),
            defect_sink_mark=int(i % 4 == 0),
            defect_burn_mark=int(i % 5 == 0),
            defect_weld_line=int(i % 6 == 0),
        ))
    return rows


@unittest.skipUnless(_PREDICTIVE_AVAILABLE, "lightgbm / scikit-learn not installed")
class TestGbtTrainer(unittest.TestCase):

    def setUp(self) -> None:
        from moldvision.predictive.trainer import GbtTrainingConfig
        self.rows = _make_rows(30)
        self.config = GbtTrainingConfig(
            n_estimators=10,   # fast for CI
            cv_folds=2,
            min_rows=5,
        )

    def test_train_returns_result(self) -> None:
        from moldvision.predictive.trainer import train_suggestion_models
        result = train_suggestion_models(self.rows, config=self.config)
        self.assertGreater(len(result.targets), 0)

    def test_feature_keys_populated(self) -> None:
        from moldvision.predictive.trainer import train_suggestion_models
        result = train_suggestion_models(self.rows, config=self.config)
        self.assertGreater(len(result.feature_keys), 0)

    def test_imputation_values_populated(self) -> None:
        from moldvision.predictive.trainer import train_suggestion_models
        result = train_suggestion_models(self.rows, config=self.config)
        for key in result.feature_keys:
            self.assertIn(key, result.imputation_values)

    def test_parameter_schema_populated(self) -> None:
        from moldvision.predictive.trainer import train_suggestion_models
        result = train_suggestion_models(self.rows, config=self.config)
        self.assertEqual(result.parameter_schema[0]["parameter_id"], "temp_barrel")

    def test_quality_score_target_trained(self) -> None:
        from moldvision.predictive.trainer import train_suggestion_models
        result = train_suggestion_models(self.rows, config=self.config)
        self.assertIn("quality_score", result.targets)

    def test_cv_metric_values_finite(self) -> None:
        from moldvision.predictive.trainer import train_suggestion_models
        import math
        result = train_suggestion_models(self.rows, config=self.config)
        for tr in result.targets.values():
            self.assertTrue(math.isfinite(tr.cv_metric_value))
            self.assertTrue(math.isfinite(tr.cv_metric_std))

    def test_eligible_only_filtering(self) -> None:
        """Rows with training_ready=False should not affect training."""
        from moldvision.predictive.trainer import train_suggestion_models
        rows = _make_rows(30)
        # Mark half as not eligible.
        for row in rows[:15]:
            row["eligibility"]["training_ready"] = False
        result = train_suggestion_models(rows, config=self.config)
        self.assertLessEqual(result.n_eligible_rows, 15)

    def test_too_few_rows_raises(self) -> None:
        from moldvision.predictive.trainer import GbtTrainingConfig, train_suggestion_models
        cfg = GbtTrainingConfig(n_estimators=5, cv_folds=2, min_rows=100)
        with self.assertRaises(ValueError):
            train_suggestion_models(self.rows, config=cfg)

    def test_nan_handling_mean_impute(self) -> None:
        """Rows with missing features should not crash mean_impute training."""
        from moldvision.predictive.trainer import GbtTrainingConfig, train_suggestion_models
        rows = _make_rows(30)
        # Remove a feature from half the rows.
        for i, row in enumerate(rows):
            if i % 2 == 0:
                row["features"].pop("injection_speed:actual.mean", None)
        cfg = GbtTrainingConfig(n_estimators=5, cv_folds=2, min_rows=5, null_strategy="mean_impute")
        result = train_suggestion_models(rows, config=cfg)
        self.assertIn("quality_score", result.targets)


@unittest.skipUnless(_PREDICTIVE_AVAILABLE, "lightgbm / scikit-learn not installed")
class TestOnnxExport(unittest.TestCase):

    def test_export_returns_bytes(self) -> None:
        try:
            import onnxmltools  # noqa: F401
        except ImportError:
            self.skipTest("onnxmltools not installed")

        from moldvision.predictive.trainer import (
            GbtTrainingConfig,
            export_target_to_onnx,
            train_suggestion_models,
        )
        rows = _make_rows(30)
        config = GbtTrainingConfig(n_estimators=5, cv_folds=2, min_rows=5)
        result = train_suggestion_models(rows, config=config)
        tr = result.targets["quality_score"]
        onnx_bytes = export_target_to_onnx(tr)
        self.assertIsInstance(onnx_bytes, bytes)
        self.assertGreater(len(onnx_bytes), 100)


if __name__ == "__main__":
    unittest.main()
