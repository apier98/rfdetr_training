"""Tests for moldvision.publish module."""
import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from moldvision.publish import (
    _read_manifest,
    _ensure_mpk,
    _sha256_file,
    publish_bundle,
    CATALOG_SCHEMA_VERSION,
)


@pytest.fixture
def sample_manifest():
    return {
        "bundle_id": "test-detector-v1.0.0",
        "model_name": "test-detector",
        "model_version": "1.0.0",
        "channel": "stable",
        "supersedes": None,
        "min_app_version": "0.1.0",
    }


@pytest.fixture
def bundle_dir(tmp_path, sample_manifest):
    """Create a minimal bundle directory."""
    d = tmp_path / "test-bundle"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(sample_manifest))
    (d / "model.onnx").write_bytes(b"fake-onnx-data")
    (d / "model_config.json").write_text('{"task": "detect"}')
    (d / "preprocess.json").write_text('{}')
    (d / "postprocess.json").write_text('{}')
    (d / "classes.json").write_text('["Component_Base"]')
    return d


@pytest.fixture
def bundle_mpk(bundle_dir):
    """Create a .mpk zip from the bundle directory."""
    mpk_path = bundle_dir.with_suffix(".mpk")
    with zipfile.ZipFile(mpk_path, "w") as zf:
        for f in sorted(bundle_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(bundle_dir))
    return mpk_path


class TestReadManifest:
    def test_from_directory(self, bundle_dir, sample_manifest):
        result = _read_manifest(bundle_dir)
        assert result["bundle_id"] == sample_manifest["bundle_id"]

    def test_from_mpk(self, bundle_mpk, sample_manifest):
        result = _read_manifest(bundle_mpk)
        assert result["bundle_id"] == sample_manifest["bundle_id"]

    def test_missing_manifest(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            _read_manifest(d)


class TestEnsureMpk:
    def test_directory_to_mpk(self, bundle_dir):
        result = _ensure_mpk(bundle_dir)
        assert result.suffix == ".mpk"
        assert result.exists()
        with zipfile.ZipFile(result) as zf:
            assert "manifest.json" in zf.namelist()

    def test_existing_mpk_passthrough(self, bundle_mpk):
        result = _ensure_mpk(bundle_mpk)
        assert result == bundle_mpk


class TestSha256File:
    def test_deterministic(self, bundle_mpk):
        h1 = _sha256_file(bundle_mpk)
        h2 = _sha256_file(bundle_mpk)
        assert h1 == h2
        assert len(h1) == 64


class TestPublishBundle:
    def test_dry_run_no_s3(self, bundle_dir):
        result = publish_bundle(
            bundle_dir,
            role="defect_detector",
            channel="stable",
            dry_run=True,
        )
        assert result["bundle_id"] == "test-detector-v1.0.0"
        assert result["role"] == "defect_detector"
        assert result["channel"] == "stable"
        assert result["compatible_layouts"] == ["*"]
        assert result["sha256"]
        assert result["size_bytes"] > 0

    def test_dry_run_custom_layouts(self, bundle_dir):
        result = publish_bundle(
            bundle_dir,
            role="defect_detector",
            compatible_layouts=["layout_A", "layout_B"],
            dry_run=True,
        )
        assert result["compatible_layouts"] == ["layout_A", "layout_B"]

    @patch("moldvision.publish._load_publish_config")
    @patch("moldvision.publish._s3_client")
    def test_publish_with_mocked_s3(self, mock_s3_client_fn, mock_config, bundle_dir):
        mock_config.return_value = {
            "bucket": "test-bucket",
            "region": "us-east-1",
            "prefix": "",
        }
        mock_s3 = MagicMock()
        mock_s3_client_fn.return_value = mock_s3
        # Mock empty catalog
        mock_s3.get_object.side_effect = Exception("NoSuchKey")

        result = publish_bundle(
            bundle_dir,
            role="defect_detector",
            channel="stable",
        )

        assert result["bundle_id"] == "test-detector-v1.0.0"
        mock_s3.upload_file.assert_called_once()
        mock_s3.put_object.assert_called_once()

        # Verify catalog was written correctly
        catalog_body = mock_s3.put_object.call_args[1]["Body"]
        catalog = json.loads(catalog_body.decode("utf-8"))
        assert catalog["schema_version"] == CATALOG_SCHEMA_VERSION
        assert len(catalog["models"]) == 1
        assert catalog["models"][0]["bundle_id"] == "test-detector-v1.0.0"

    @patch("moldvision.publish._load_publish_config")
    @patch("moldvision.publish._s3_client")
    def test_publish_updates_existing_catalog(self, mock_s3_client_fn, mock_config, bundle_dir):
        mock_config.return_value = {"bucket": "b", "region": "r", "prefix": ""}
        mock_s3 = MagicMock()
        mock_s3_client_fn.return_value = mock_s3

        existing_catalog = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "updated_at": "2026-01-01T00:00:00",
            "models": [
                {"bundle_id": "old-model-v0.9.0", "role": "defect_detector"},
            ],
        }
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(existing_catalog).encode())
        }

        result = publish_bundle(bundle_dir, role="defect_detector")

        catalog_body = mock_s3.put_object.call_args[1]["Body"]
        catalog = json.loads(catalog_body.decode("utf-8"))
        assert len(catalog["models"]) == 2
        bundle_ids = [m["bundle_id"] for m in catalog["models"]]
        assert "old-model-v0.9.0" in bundle_ids
        assert "test-detector-v1.0.0" in bundle_ids

    def test_publish_to_shared_root_writes_bundle_directory_and_index(self, bundle_dir, tmp_path, monkeypatch):
        shared_root = tmp_path / "shared"
        monkeypatch.setenv("ARIA_SHARED_ROOT", str(shared_root))
        monkeypatch.setenv("ARIA_MODEL_PUBLISH_TARGET", "shared")

        result = publish_bundle(bundle_dir, role="defect_detector")

        published_root = shared_root / "published" / "moldpilot" / "detection"
        bundle_copy = published_root / "bundles" / "test-detector-v1.0.0"
        assert result["publish_target"] == "shared"
        assert bundle_copy.exists()
        assert (bundle_copy / "manifest.json").exists()
        index = json.loads((published_root / "index.json").read_text(encoding="utf-8"))
        assert index["active_by_channel"]["stable"] == "test-detector-v1.0.0"
        assert index["bundles"][0]["artifact_key"] == "bundles/test-detector-v1.0.0/"
