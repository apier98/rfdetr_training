"""Tests for finetune-from bundle resolution."""
import json
import os
from pathlib import Path

import pytest


def _make_registry(tmp_path, task_dir="defect_detection"):
    """Create a minimal lake registry with one bundle."""
    lake_root = tmp_path / "lake"
    models_dir = lake_root / "models" / task_dir
    models_dir.mkdir(parents=True)

    bundle_dir = models_dir / "test-detector-v1.0.0"
    bundle_dir.mkdir()
    (bundle_dir / "checkpoint_portable.pth").write_bytes(b"fake-weights")
    (bundle_dir / "manifest.json").write_text(json.dumps({
        "bundle_id": "test-detector-v1.0.0",
        "model_version": "1.0.0",
    }))

    registry = {
        "task": "detect",
        "bundles": [
            {
                "bundle_id": "test-detector-v1.0.0",
                "version": "1.0.0",
                "channel": "stable",
                "path": str(bundle_dir),
            }
        ],
        "active": {
            "stable": "test-detector-v1.0.0",
        },
    }
    (models_dir / "registry.json").write_text(json.dumps(registry))
    return lake_root


class TestFinetuneResolve:
    def test_resolve_by_bundle_id(self, tmp_path):
        from moldvision.cli_handlers import _resolve_finetune_weights
        from types import SimpleNamespace

        lake_root = _make_registry(tmp_path)
        config_path = tmp_path / "data_lake_config.json"
        config_path.write_text(json.dumps({"root": str(lake_root)}))

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            args = SimpleNamespace(task="detect", lake_root=str(lake_root))
            result = _resolve_finetune_weights("test-detector-v1.0.0", args)
            assert "checkpoint_portable.pth" in result
            assert Path(result).exists()
        finally:
            os.chdir(old_cwd)

    def test_resolve_latest(self, tmp_path):
        from moldvision.cli_handlers import _resolve_finetune_weights
        from types import SimpleNamespace

        lake_root = _make_registry(tmp_path)
        config_path = tmp_path / "data_lake_config.json"
        config_path.write_text(json.dumps({"root": str(lake_root)}))

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            args = SimpleNamespace(task="detect", lake_root=str(lake_root))
            result = _resolve_finetune_weights("latest", args)
            assert "checkpoint_portable.pth" in result
        finally:
            os.chdir(old_cwd)

    def test_resolve_unknown_bundle(self, tmp_path):
        from moldvision.cli_handlers import _resolve_finetune_weights
        from types import SimpleNamespace

        lake_root = _make_registry(tmp_path)
        config_path = tmp_path / "data_lake_config.json"
        config_path.write_text(json.dumps({"root": str(lake_root)}))

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            args = SimpleNamespace(task="detect", lake_root=str(lake_root))
            with pytest.raises(ValueError, match="not found"):
                _resolve_finetune_weights("nonexistent-bundle", args)
        finally:
            os.chdir(old_cwd)
