"""Tests for the startup-suggestion bundle writer and packer.

Requires ``[predictive]`` optional deps.  Skipped otherwise.
"""
from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    import lightgbm  # noqa: F401
    import numpy  # noqa: F401
    _PREDICTIVE_AVAILABLE = True
except ImportError:
    _PREDICTIVE_AVAILABLE = False

try:
    import onnxmltools  # noqa: F401
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


def _make_row(session_id: str, i: int) -> dict:
    return {
        "schema_version": "training_row_v1",
        "session_id": session_id,
        "component_id": f"cmp_{i % 5}",
        "eligibility": {"training_ready": True, "base_quality_gate_ready": True, "coverage_ratio": 0.9},
        "features": {
            "temp_barrel:actual.mean": 210.0 + i,
            "pressure_injection:actual.last": 800.0 + i * 2,
        },
        "targets": {
            "y_quality_score": 0.5 + (i % 10) * 0.04,
            "y_defect_flash": int(i % 3 == 0),
            "y_defect_sink_mark": int(i % 4 == 0),
            "y_defect_burn_mark": int(i % 5 == 0),
            "y_defect_weld_line": int(i % 6 == 0),
        },
        "context": {
            "defect_classes_monitored": ["flash", "sink_mark"],
            "feature_keys": ["temp_barrel:actual.mean", "pressure_injection:actual.last"],
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


@unittest.skipUnless(
    _PREDICTIVE_AVAILABLE and _ONNX_AVAILABLE,
    "lightgbm / onnxmltools not installed",
)
class TestWriteSuggestionBundle(unittest.TestCase):

    def _train(self, n: int = 30):
        from moldvision.predictive.trainer import GbtTrainingConfig, train_suggestion_models
        rows = [_make_row(f"sess_{i:03d}", i) for i in range(n)]
        cfg = GbtTrainingConfig(n_estimators=5, cv_folds=2, min_rows=5)
        return train_suggestion_models(rows, config=cfg)

    def test_bundle_dir_created(self) -> None:
        from moldvision.predictive.suggestion_bundle import write_suggestion_bundle
        result = self._train()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = write_suggestion_bundle(
                Path(td), result, model_name="startup-suggestion", model_version="0.0.1"
            )
            self.assertTrue(bundle_dir.is_dir())

    def test_manifest_json_present(self) -> None:
        from moldvision.predictive.suggestion_bundle import write_suggestion_bundle
        result = self._train()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = write_suggestion_bundle(
                Path(td), result, model_name="startup-suggestion", model_version="0.0.1"
            )
            manifest_path = bundle_dir / "manifest.json"
            self.assertTrue(manifest_path.exists())

    def test_manifest_bundle_type(self) -> None:
        from moldvision.predictive.suggestion_bundle import write_suggestion_bundle
        result = self._train()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = write_suggestion_bundle(
                Path(td), result, model_name="startup-suggestion", model_version="0.0.1"
            )
            manifest = json.loads((bundle_dir / "manifest.json").read_text())
            self.assertEqual(manifest["bundle_type"], "startup_suggestion")

    def test_manifest_feature_keys_match(self) -> None:
        from moldvision.predictive.suggestion_bundle import write_suggestion_bundle
        result = self._train()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = write_suggestion_bundle(
                Path(td), result, model_name="startup-suggestion", model_version="0.0.1"
            )
            manifest = json.loads((bundle_dir / "manifest.json").read_text())
            self.assertEqual(manifest["feature_keys"], result.feature_keys)

    def test_manifest_parameter_schema_match(self) -> None:
        from moldvision.predictive.suggestion_bundle import write_suggestion_bundle
        result = self._train()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = write_suggestion_bundle(
                Path(td), result, model_name="startup-suggestion", model_version="0.0.1"
            )
            manifest = json.loads((bundle_dir / "manifest.json").read_text())
            self.assertEqual(manifest["parameter_schema"], result.parameter_schema)

    def test_onnx_files_present(self) -> None:
        from moldvision.predictive.suggestion_bundle import write_suggestion_bundle
        result = self._train()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = write_suggestion_bundle(
                Path(td), result, model_name="startup-suggestion", model_version="0.0.1"
            )
            manifest = json.loads((bundle_dir / "manifest.json").read_text())
            for filename in manifest["target_models"].values():
                self.assertTrue(
                    (bundle_dir / filename).exists(),
                    f"ONNX file missing: {filename}",
                )

    def test_checksums_correct(self) -> None:
        import hashlib
        from moldvision.predictive.suggestion_bundle import write_suggestion_bundle
        result = self._train()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = write_suggestion_bundle(
                Path(td), result, model_name="startup-suggestion", model_version="0.0.1"
            )
            manifest = json.loads((bundle_dir / "manifest.json").read_text())
            for filename, expected in manifest["checksums"].items():
                h = hashlib.sha256((bundle_dir / filename).read_bytes()).hexdigest()
                self.assertEqual(h, expected, f"Checksum mismatch for {filename}")

    def test_training_meta_json_present(self) -> None:
        from moldvision.predictive.suggestion_bundle import write_suggestion_bundle
        result = self._train()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = write_suggestion_bundle(
                Path(td), result, model_name="startup-suggestion", model_version="0.0.1"
            )
            meta_path = bundle_dir / "training_meta.json"
            self.assertTrue(meta_path.exists())
            meta = json.loads(meta_path.read_text())
            self.assertIn("n_eligible_rows", meta)
            self.assertIn("cv_metrics", meta)
            self.assertEqual(meta["null_strategy"], "native_missing")
            self.assertIn("min_feature_presence_ratio", meta["lgbm_config"])

    def test_pack_sugbundle_creates_zip(self) -> None:
        from moldvision.predictive.suggestion_bundle import pack_sugbundle, write_suggestion_bundle
        result = self._train()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = write_suggestion_bundle(
                Path(td), result, model_name="startup-suggestion", model_version="0.0.1"
            )
            archive = pack_sugbundle(bundle_dir)
            self.assertTrue(archive.exists())
            self.assertEqual(archive.suffix, ".sugbundle")
            self.assertTrue(zipfile.is_zipfile(archive))

    def test_pack_contains_manifest(self) -> None:
        from moldvision.predictive.suggestion_bundle import pack_sugbundle, write_suggestion_bundle
        result = self._train()
        with tempfile.TemporaryDirectory() as td:
            bundle_dir = write_suggestion_bundle(
                Path(td), result, model_name="startup-suggestion", model_version="0.0.1"
            )
            archive = pack_sugbundle(bundle_dir)
            with zipfile.ZipFile(archive, "r") as zf:
                names = zf.namelist()
            self.assertTrue(
                any("manifest.json" in n for n in names),
                f"manifest.json not in archive: {names}",
            )


if __name__ == "__main__":
    unittest.main()
