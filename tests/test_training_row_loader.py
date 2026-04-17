"""Tests for the predictive training_row_loader module."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from moldvision.predictive.training_row_loader import (
    assess_training_readiness,
    check_schema_homogeneity,
    extract_parameter_schema,
    extract_feature_matrix,
    extract_targets,
    filter_eligible,
    infer_feature_keys,
    load_training_rows,
    summarize_scope_distribution,
    summarize_dataset,
    validate_dataset,
    validate_row,
)


def _make_row(
    *,
    session_id: str = "sess_001",
    component_id: str = "cmp_1",
    training_ready: bool = True,
    features: dict | None = None,
    targets: dict | None = None,
    feature_keys: list[str] | None = None,
    hmi_layout_id: str | None = "layout_A",
    machine_family: str | None = "FAMILY_A",
    parameter_schema: list[dict] | None = None,
) -> dict:
    """Build a minimal valid training_row_v1 record."""
    if features is None:
        features = {
            "temp_barrel:actual.last": 221.0,
            "temp_barrel:actual.mean": 220.5,
            "pressure_injection:actual.last": 850.0,
            "pressure_injection:actual.mean": 848.0,
        }
    if targets is None:
        targets = {
            "y_quality_score": 0.85,
            "y_defect_flash": 0,
            "y_defect_sink_mark": 1,
            "y_burden_sink_mark": 0.15,
            "labels_present": ["sink_mark"],
            "quality_formula": {
                "type": "weighted_burden_complement_v1",
                "weights": {"sink_mark": 0.35, "flash": 0.30},
            },
        }
    if feature_keys is None:
        feature_keys = sorted(features.keys())

    return {
        "schema_version": "training_row_v1",
        "created_utc": "2025-01-01T00:00:00Z",
        "run_id": "run_1",
        "session_id": session_id,
        "component_id": component_id,
        "traceability": {"company": "TEST"},
        "eligibility": {
            "training_ready": training_ready,
            "base_quality_gate_ready": training_ready,
            "has_params": True,
            "coverage_ratio": 0.85,
            "has_blocking_flags": False,
            "blocking_flags": [],
            "recording_mode_ok": True,
            "require_qualification": True,
            "critical_slots_present": 2,
            "critical_slots_required": 2,
            "critical_slots_ok": True,
            "critical_slot_patterns": ["temp_barrel", "pressure_injection"],
            "critical_slot_matches": ["temp_barrel:actual", "pressure_injection:actual"],
        },
        "window": {
            "start_t": 10.0,
            "end_t": 12.0,
            "process_video_id": "proc_01",
            "process_video_ids": ["proc_01"],
            "inspection_video_id": "insp_01",
        },
        "features": features,
        "targets": targets,
        "context": {
            "qa_flags": [],
            "defect_labels_detected": ["Sink_Mark"],
            "defect_classes_monitored": ["burn_mark", "flash", "sink_mark", "weld_line"],
            "hmi_layout_id": hmi_layout_id,
            "hmi_layout_version": "v1",
            "machine_family": machine_family,
            "feature_keys": feature_keys,
            "parameter_schema": parameter_schema or [],
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


class TestLoadTrainingRows(unittest.TestCase):
    def test_load_valid_file(self) -> None:
        rows = [_make_row(), _make_row(session_id="sess_002")]
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            loaded = load_training_rows(Path(f.name))
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["session_id"], "sess_001")

    def test_load_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            f.write("\n\n")
            f.flush()
            loaded = load_training_rows(Path(f.name))
        self.assertEqual(loaded, [])

    def test_load_malformed_raises(self) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            f.write("{bad json\n")
            f.flush()
            with self.assertRaises(ValueError) as ctx:
                load_training_rows(Path(f.name))
            self.assertIn("line 1", str(ctx.exception))


class TestValidateRow(unittest.TestCase):
    def test_valid_row_has_no_errors(self) -> None:
        errors = validate_row(_make_row())
        self.assertEqual(errors, [])

    def test_wrong_schema_version(self) -> None:
        row = _make_row()
        row["schema_version"] = "training_row_v99"
        errors = validate_row(row)
        self.assertTrue(any("schema_version" in e for e in errors))

    def test_v2_schema_accepted(self) -> None:
        row = _make_row()
        row["schema_version"] = "training_row_v2"
        errors = validate_row(row)
        self.assertFalse(any("schema_version" in e for e in errors))

    def test_missing_features(self) -> None:
        row = _make_row()
        del row["features"]
        errors = validate_row(row)
        self.assertTrue(any("features" in e for e in errors))

    def test_empty_features_reported(self) -> None:
        row = _make_row(features={})
        errors = validate_row(row)
        self.assertTrue(any("empty" in e for e in errors))

    def test_missing_targets_key(self) -> None:
        row = _make_row()
        del row["targets"]["y_quality_score"]
        errors = validate_row(row)
        self.assertTrue(any("y_quality_score" in e for e in errors))

    def test_missing_context_keys(self) -> None:
        row = _make_row()
        del row["context"]["feature_keys"]
        errors = validate_row(row)
        self.assertTrue(any("feature_keys" in e for e in errors))


class TestValidateDataset(unittest.TestCase):
    def test_all_valid(self) -> None:
        report = validate_dataset([_make_row(), _make_row()])
        self.assertTrue(report["valid"])
        self.assertEqual(report["invalid_rows"], 0)

    def test_some_invalid(self) -> None:
        good = _make_row()
        bad = _make_row()
        del bad["features"]
        report = validate_dataset([good, bad])
        self.assertFalse(report["valid"])
        self.assertEqual(report["invalid_rows"], 1)

    def test_scope_warnings_capture_missing_scope_fields(self) -> None:
        row = _make_row()
        row["mold_id"] = None
        row["material_id"] = None
        report = validate_dataset([row])
        self.assertEqual(report["scope_warnings"]["null_mold_id"], 1)
        self.assertEqual(report["scope_warnings"]["null_material_id"], 1)


class TestFilterEligible(unittest.TestCase):
    def test_filters_ineligible(self) -> None:
        eligible = _make_row(training_ready=True)
        ineligible = _make_row(training_ready=False)
        result = filter_eligible([eligible, ineligible])
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["eligibility"]["training_ready"])


class TestFeatureExtraction(unittest.TestCase):
    def test_infer_feature_keys_from_context(self) -> None:
        row = _make_row()
        keys = infer_feature_keys([row])
        self.assertIn("temp_barrel:actual.last", keys)
        self.assertIn("pressure_injection:actual.last", keys)

    def test_extract_matrix_shape(self) -> None:
        rows = [_make_row(), _make_row(session_id="sess_002")]
        matrix, cols = extract_feature_matrix(rows)
        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(matrix[0]), len(cols))

    def test_extract_matrix_missing_values_are_none(self) -> None:
        row1 = _make_row(features={"a.last": 1.0, "b.last": 2.0})
        row2 = _make_row(features={"a.last": 3.0})
        row1["context"]["feature_keys"] = ["a.last", "b.last"]
        row2["context"]["feature_keys"] = ["a.last"]
        matrix, cols = extract_feature_matrix([row1, row2])
        self.assertEqual(cols, ["a.last", "b.last"])
        self.assertEqual(matrix[0], [1.0, 2.0])
        self.assertEqual(matrix[1], [3.0, None])

    def test_extract_matrix_with_explicit_keys(self) -> None:
        row = _make_row()
        matrix, cols = extract_feature_matrix([row], feature_keys=["temp_barrel:actual.last"])
        self.assertEqual(cols, ["temp_barrel:actual.last"])
        self.assertEqual(len(matrix[0]), 1)
        self.assertEqual(matrix[0][0], 221.0)

    def test_extract_parameter_schema_merged_and_filtered(self) -> None:
        rows = [
            _make_row(
                parameter_schema=[
                    {
                        "parameter_id": "temp_barrel:zone_1",
                        "page_id": "temperature",
                        "subpage_id": "barrel",
                        "canonical_parameter_id": "temp_barrel",
                        "slot_id": "zone_1",
                        "canonical_slot_id": "zone_1",
                        "display_name": "Barrel Temperature",
                        "unit": "C",
                        "baseline": 220.0,
                        "range_min": 180.0,
                        "range_max": 320.0,
                        "control_feature_keys": [
                            "temp_barrel:zone_1.setpoint_end",
                            "temp_barrel:zone_1.setpoint",
                        ],
                        "step_mode": "absolute",
                        "preferred_step": 1.0,
                        "max_delta": 5.0,
                    }
                ]
            )
        ]

        schema = extract_parameter_schema(rows, feature_keys=["temp_barrel:zone_1.setpoint_end"])

        self.assertEqual(len(schema), 1)
        self.assertEqual(schema[0]["parameter_id"], "temp_barrel:zone_1")
        self.assertEqual(schema[0]["page_id"], "temperature")
        self.assertEqual(schema[0]["subpage_id"], "barrel")
        self.assertEqual(schema[0]["canonical_parameter_id"], "temp_barrel")
        self.assertEqual(schema[0]["slot_id"], "zone_1")
        self.assertEqual(schema[0]["control_feature_keys"], ["temp_barrel:zone_1.setpoint_end"])


class TestExtractTargets(unittest.TestCase):
    def test_quality_score(self) -> None:
        rows = [_make_row()]
        targets = extract_targets(rows, "y_quality_score")
        self.assertEqual(targets, [0.85])

    def test_missing_target_is_none(self) -> None:
        row = _make_row()
        del row["targets"]["y_quality_score"]
        targets = extract_targets([row], "y_quality_score")
        self.assertEqual(targets, [None])

    def test_defect_binary(self) -> None:
        targets = extract_targets([_make_row()], "y_defect_sink_mark")
        self.assertEqual(targets, [1.0])


class TestSchemaHomogeneity(unittest.TestCase):
    def test_homogeneous(self) -> None:
        rows = [_make_row(), _make_row(session_id="sess_002")]
        result = check_schema_homogeneity(rows)
        self.assertTrue(result["homogeneous"])
        self.assertEqual(result["n_schemas"], 1)

    def test_same_layout_with_different_active_slots_is_still_homogeneous(self) -> None:
        row1 = _make_row(feature_keys=["a.setpoint_end", "a.present"], features={"a.setpoint_end": 1.0, "a.present": 1.0})
        row2 = _make_row(
            session_id="sess_002",
            feature_keys=["a.setpoint_end", "a.present", "b.setpoint_end", "b.present"],
            features={
                "a.setpoint_end": 1.0,
                "a.present": 1.0,
                "b.setpoint_end": 2.0,
                "b.present": 1.0,
            },
        )

        result = check_schema_homogeneity([row1, row2])

        self.assertTrue(result["homogeneous"])
        self.assertEqual(result["n_schemas"], 1)

    def test_heterogeneous(self) -> None:
        row1 = _make_row()
        row2 = _make_row(
            session_id="sess_002",
            features={"x.last": 1.0},
            feature_keys=["x.last"],
            hmi_layout_id="layout_B",
            machine_family="FAMILY_B",
        )
        result = check_schema_homogeneity([row1, row2])
        self.assertFalse(result["homogeneous"])
        self.assertEqual(result["n_schemas"], 2)


class TestScopeDistribution(unittest.TestCase):
    def test_scope_distribution_lists_distinct_scopes_and_families(self) -> None:
        row1 = _make_row(machine_family="FAMILY_A")
        row1["mold_id"] = "mold_a12"
        row1["material_id"] = "pp"
        row2 = _make_row(session_id="sess_002", machine_family="FAMILY_B")
        row2["mold_id"] = "mold_a12"
        row2["material_id"] = "abs"

        result = summarize_scope_distribution([row1, row2])

        self.assertEqual(result["distinct_scope_count"], 2)
        self.assertEqual(result["machine_family_count"], 2)
        self.assertIn(("mold_a12", "pp"), result["distinct_scopes"])
        self.assertIn("FAMILY_A", result["machine_families"])


class TestTrainingReadiness(unittest.TestCase):
    def test_readiness_thresholds(self) -> None:
        self.assertEqual(assess_training_readiness(5)["level"], "blocked")
        self.assertEqual(assess_training_readiness(25)["level"], "poor")
        self.assertEqual(assess_training_readiness(75)["level"], "weak")
        self.assertEqual(assess_training_readiness(200)["level"], "good")
        self.assertEqual(assess_training_readiness(400)["level"], "strong")


class TestSummarizeDataset(unittest.TestCase):
    def test_summary_structure(self) -> None:
        rows = [_make_row(), _make_row(session_id="sess_002", training_ready=False)]
        summary = summarize_dataset(rows)
        self.assertEqual(summary["total_rows"], 2)
        self.assertEqual(summary["eligible_rows"], 1)
        self.assertIn("feature_columns", summary)
        self.assertIn("quality_score_stats", summary)
        self.assertIn("defect_counts", summary)
        self.assertIn("sink_mark", summary["defect_counts"])
        self.assertIn("training_readiness", summary)


if __name__ == "__main__":
    unittest.main()
