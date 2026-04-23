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
            "y_burden_flash": 0.6 if defect_flash else 0.0,
            "y_burden_sink_mark": 0.6 if defect_sink_mark else 0.0,
            "y_burden_burn_mark": 0.6 if defect_burn_mark else 0.0,
            "y_burden_weld_line": 0.6 if defect_weld_line else 0.0,
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
                    "family_id": "temp_barrel",
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
        self.assertGreater(len(result.targets["quality_score"].used_feature_keys), 0)

    def test_imputation_values_populated(self) -> None:
        from moldvision.predictive.trainer import train_suggestion_models
        result = train_suggestion_models(self.rows, config=self.config)
        for key in result.feature_keys:
            self.assertIn(key, result.imputation_values)
        self.assertEqual(result.null_strategy, "native_missing")

    def test_parameter_schema_populated(self) -> None:
        from moldvision.predictive.trainer import train_suggestion_models
        result = train_suggestion_models(self.rows, config=self.config)
        self.assertEqual(result.parameter_schema[0]["parameter_id"], "temp_barrel")

    def test_quality_score_target_trained(self) -> None:
        from moldvision.predictive.trainer import train_suggestion_models
        result = train_suggestion_models(self.rows, config=self.config)
        self.assertIn("quality_score", result.targets)

    def test_defect_targets_use_regression_burden_signals(self) -> None:
        from moldvision.predictive.trainer import train_suggestion_models

        result = train_suggestion_models(self.rows, config=self.config)

        defect_target = result.targets["defect_flash"]
        self.assertEqual(defect_target.model_type, "regression")
        self.assertEqual(defect_target.source_target, "y_burden_flash")
        self.assertEqual(defect_target.signal_kind, "defect_burden")
        self.assertEqual(defect_target.cv_metric_name, "rmse")

    def test_constant_regression_target_is_skipped(self) -> None:
        from moldvision.predictive.trainer import train_suggestion_models

        rows = _make_rows(30)
        for row in rows:
            row["targets"]["y_burden_flash"] = 0.0

        result = train_suggestion_models(rows, config=self.config)

        self.assertIn("quality_score", result.targets)
        self.assertNotIn("defect_flash", result.targets)
        self.assertIn("defect_sink_mark", result.targets)

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

    def test_nan_handling_native_missing(self) -> None:
        """Rows with missing features should not crash native-missing training."""
        from moldvision.predictive.trainer import GbtTrainingConfig, train_suggestion_models
        rows = _make_rows(30)
        for i, row in enumerate(rows):
            if i % 2 == 0:
                row["features"].pop("injection_speed:actual.mean", None)
                row["context"]["feature_keys"] = [
                    key for key in row["context"]["feature_keys"]
                    if key != "injection_speed:actual.mean"
                ]
        cfg = GbtTrainingConfig(n_estimators=5, cv_folds=2, min_rows=5, null_strategy="native_missing")
        result = train_suggestion_models(rows, config=cfg)
        self.assertIn("quality_score", result.targets)
        self.assertEqual(result.null_strategy, "native_missing")

    def test_sparse_feature_pruning_drops_rare_columns(self) -> None:
        from moldvision.predictive.trainer import GbtTrainingConfig, train_suggestion_models
        rows = _make_rows(30)
        rows[0]["features"]["rare_slot:actual.mean"] = 42.0
        rows[0]["context"]["feature_keys"].append("rare_slot:actual.mean")
        cfg = GbtTrainingConfig(
            n_estimators=5,
            cv_folds=2,
            min_rows=5,
            min_feature_presence_ratio=0.10,
        )
        result = train_suggestion_models(rows, config=cfg)
        self.assertNotIn("rare_slot:actual.mean", result.feature_keys)

    def test_selected_feature_stats_keep_only_setpoint_end_and_present(self) -> None:
        from moldvision.predictive.trainer import GbtTrainingConfig, train_suggestion_models
        rows = []
        for i in range(12):
            rows.append(
                {
                    "schema_version": "training_row_v2",
                    "session_id": f"sess_{i:03d}",
                    "component_id": f"cmp_{i:03d}",
                    "eligibility": {
                        "training_ready": True,
                        "base_quality_gate_ready": True,
                        "coverage_ratio": 0.95,
                    },
                    "features": {
                        "pressure_injection:step_1.setpoint_end": 800.0 + i,
                        "pressure_injection:step_1.present": 1.0,
                        "pressure_injection:step_1.changed": float(i % 2),
                        "pressure_injection:step_1.n_changes": float(i % 3),
                    },
                    "targets": {
                        "y_quality_score": 0.5 + i * 0.01,
                        "y_defect_flash": int(i % 3 == 0),
                        "y_defect_sink_mark": 0,
                        "y_defect_burn_mark": 0,
                        "y_defect_weld_line": 0,
                        "y_burden_flash": 0.6 if i % 3 == 0 else 0.0,
                        "y_burden_sink_mark": 0.0,
                        "y_burden_burn_mark": 0.0,
                        "y_burden_weld_line": 0.0,
                    },
                    "context": {
                        "defect_classes_monitored": ["flash"],
                        "feature_keys": [
                            "pressure_injection:step_1.setpoint_end",
                            "pressure_injection:step_1.present",
                            "pressure_injection:step_1.changed",
                            "pressure_injection:step_1.n_changes",
                        ],
                        "parameter_schema": [
                            {
                                "parameter_id": "pressure_injection:step_1",
                                "display_name": "Injection Pressure - Step 1",
                                "unit": "bar",
                                "baseline": 800.0,
                                "range_min": 500.0,
                                "range_max": 1200.0,
                                "control_feature_keys": [
                                    "pressure_injection:step_1.setpoint_end",
                                ],
                                "step_mode": "absolute",
                                "preferred_step": 10.0,
                                "max_delta": 50.0,
                            }
                        ],
                    },
                }
            )
        cfg = GbtTrainingConfig(n_estimators=5, cv_folds=2, min_rows=5)
        result = train_suggestion_models(rows, config=cfg)
        self.assertEqual(
            result.feature_keys,
            [
                "pressure_injection:step_1.setpoint_end",
            ],
        )
        self.assertEqual(
            result.parameter_schema[0]["control_feature_keys"],
            ["pressure_injection:step_1.setpoint_end"],
        )

    def test_tiny_dataset_trains_on_full_eligible_rows(self) -> None:
        from moldvision.predictive.trainer import GbtTrainingConfig, train_suggestion_models

        rows = _make_rows(11)
        cfg = GbtTrainingConfig(n_estimators=5, cv_folds=5, min_rows=5)

        result = train_suggestion_models(rows, config=cfg)

        self.assertIn("quality_score", result.targets)
        self.assertEqual(result.targets["quality_score"].n_train, 11)
        self.assertEqual(result.targets["quality_score"].n_eval, 0)

    def test_full_parameter_schema_retained_when_feature_not_trainable(self) -> None:
        from moldvision.predictive.trainer import GbtTrainingConfig, train_suggestion_models

        rows = []
        for i in range(12):
            rows.append(
                {
                    "schema_version": "training_row_v2",
                    "session_id": f"sess_{i:03d}",
                    "component_id": f"cmp_{i:03d}",
                    "eligibility": {
                        "training_ready": True,
                        "base_quality_gate_ready": True,
                        "coverage_ratio": 0.95,
                    },
                    "features": {
                        "pressure_injection:step_1.setpoint_end": 800.0 + i,
                        "temp_barrel:zone_1.setpoint_end": 220.0,
                    },
                    "targets": {
                        "y_quality_score": 0.5 + i * 0.01,
                        "y_defect_flash": int(i % 3 == 0),
                        "y_defect_sink_mark": 0,
                        "y_defect_burn_mark": 0,
                        "y_defect_weld_line": 0,
                        "y_burden_flash": 0.6 if i % 3 == 0 else 0.0,
                        "y_burden_sink_mark": 0.0,
                        "y_burden_burn_mark": 0.0,
                        "y_burden_weld_line": 0.0,
                    },
                    "context": {
                        "defect_classes_monitored": ["flash"],
                        "feature_keys": [
                            "pressure_injection:step_1.setpoint_end",
                            "temp_barrel:zone_1.setpoint_end",
                        ],
                        "parameter_schema": [
                            {
                                "parameter_id": "pressure_injection:step_1",
                                "family_id": "pressure_injection",
                                "display_name": "Injection Pressure - Step 1",
                                "unit": "bar",
                                "baseline": 800.0,
                                "range_min": 500.0,
                                "range_max": 1200.0,
                                "control_feature_keys": [
                                    "pressure_injection:step_1.setpoint_end",
                                ],
                            },
                            {
                                "parameter_id": "temp_barrel:zone_1",
                                "family_id": "temp_barrel",
                                "display_name": "Barrel Zone 1",
                                "unit": "C",
                                "baseline": 220.0,
                                "range_min": 180.0,
                                "range_max": 320.0,
                                "control_feature_keys": [
                                    "temp_barrel:zone_1.setpoint_end",
                                ],
                            },
                        ],
                    },
                }
            )

        result = train_suggestion_models(rows, config=GbtTrainingConfig(n_estimators=5, cv_folds=2, min_rows=5))

        self.assertEqual(result.feature_keys, ["pressure_injection:step_1.setpoint_end"])
        schema_by_id = {item["parameter_id"]: item for item in result.parameter_schema}
        self.assertEqual(
            schema_by_id["pressure_injection:step_1"]["trained_control_feature_keys"],
            ["pressure_injection:step_1.setpoint_end"],
        )
        self.assertEqual(
            schema_by_id["temp_barrel:zone_1"]["trained_control_feature_keys"],
            [],
        )

    def test_atomic_control_family_becomes_partially_controllable_when_safe_subset_survives(self) -> None:
        from moldvision.predictive.trainer import GbtTrainingConfig, train_suggestion_models

        rows = []
        for i in range(12):
            rows.append(
                {
                    "schema_version": "training_row_v2",
                    "session_id": f"sess_atomic_{i:03d}",
                    "component_id": f"cmp_atomic_{i:03d}",
                    "eligibility": {
                        "training_ready": True,
                        "base_quality_gate_ready": True,
                        "coverage_ratio": 0.95,
                    },
                    "features": {
                        "pressure_injection:step_1.setpoint_end": 800.0 + i,
                        "pressure_injection:step_2.setpoint_end": 700.0,
                    },
                    "targets": {
                        "y_quality_score": 0.6 + i * 0.01,
                        "y_defect_flash": int(i % 3 == 0),
                        "y_defect_sink_mark": 0,
                        "y_defect_burn_mark": 0,
                        "y_defect_weld_line": 0,
                        "y_burden_flash": 0.6 if i % 3 == 0 else 0.0,
                        "y_burden_sink_mark": 0.0,
                        "y_burden_burn_mark": 0.0,
                        "y_burden_weld_line": 0.0,
                    },
                    "context": {
                        "defect_classes_monitored": ["flash"],
                        "feature_keys": [
                            "pressure_injection:step_1.setpoint_end",
                            "pressure_injection:step_2.setpoint_end",
                        ],
                        "parameter_schema": [
                            {
                                "parameter_id": "pressure_injection:step_1",
                                "family_id": "pressure_injection",
                                "canonical_parameter_id": "pressure_injection",
                                "canonical_slot_id": "step_1",
                                "display_name": "Injection Pressure - Step 1",
                                "unit": "bar",
                                "baseline": 800.0,
                                "range_min": 500.0,
                                "range_max": 1200.0,
                                "control_feature_keys": ["pressure_injection:step_1.setpoint_end"],
                            },
                            {
                                "parameter_id": "pressure_injection:step_2",
                                "family_id": "pressure_injection",
                                "canonical_parameter_id": "pressure_injection",
                                "canonical_slot_id": "step_2",
                                "display_name": "Injection Pressure - Step 2",
                                "unit": "bar",
                                "baseline": 700.0,
                                "range_min": 500.0,
                                "range_max": 1200.0,
                                "control_feature_keys": ["pressure_injection:step_2.setpoint_end"],
                            },
                        ],
                        "control_families": [
                            {
                                "family_id": "pressure_injection",
                                "family_type": "atomic",
                                "ordered_members": [
                                    {
                                        "parameter_id": "pressure_injection:step_1",
                                        "canonical_slot_id": "step_1",
                                        "control_feature_keys": ["pressure_injection:step_1.setpoint_end"],
                                    },
                                    {
                                        "parameter_id": "pressure_injection:step_2",
                                        "canonical_slot_id": "step_2",
                                        "control_feature_keys": ["pressure_injection:step_2.setpoint_end"],
                                    },
                                ],
                            }
                        ],
                    },
                }
            )

        result = train_suggestion_models(rows, config=GbtTrainingConfig(n_estimators=5, cv_folds=2, min_rows=5))

        self.assertEqual(result.feature_keys, ["pressure_injection:step_1.setpoint_end"])
        self.assertEqual(len(result.control_families), 1)
        self.assertEqual(result.control_families[0]["deployable"], True)
        self.assertEqual(result.control_families[0]["family_type"], "partially_controllable")
        self.assertEqual(result.control_families[0]["deployability_reason"], "partial_contiguous_subset")
        self.assertEqual(
            result.control_families[0]["family_constraints"]["controllable_member_parameter_ids"],
            ["pressure_injection:step_1"],
        )
        self.assertEqual(len(result.deployable_control_families), 1)
        self.assertEqual(
            [member["parameter_id"] for member in result.deployable_control_families[0]["ordered_members"]],
            ["pressure_injection:step_1"],
        )


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
