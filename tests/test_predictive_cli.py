from __future__ import annotations

import json
import pickle
import types
from pathlib import Path
from unittest.mock import patch


def test_predictive_list_artifacts_reports_shared_exports(tmp_path, capsys) -> None:
    from moldvision.cli_handlers import _handle_predictive_list_artifacts

    export_dir = (
        tmp_path
        / "shared"
        / "ingest"
        / "moldtrace"
        / "training_rows"
        / "v1"
        / "session-123"
        / "20260416T100628Z"
    )
    export_dir.mkdir(parents=True)
    (export_dir / "export_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "moldtrace_training_rows_export_v1",
                "session_id": "session-123",
                "export_id": "20260416T100628Z",
                "staged_at_utc": "2026-04-16T10:35:22Z",
            }
        ),
        encoding="utf-8",
    )
    (export_dir / "session.json").write_text(
        json.dumps({"mold_id": "stampo_giotto", "material_id": "pc", "machine_id": "PRESSA_TEST"}),
        encoding="utf-8",
    )
    (export_dir / "training_rows_summary.json").write_text(
        json.dumps(
            {
                "counts": {"rows_total": 11, "rows_training_ready": 9},
                "critical_slot_policy": {"machine_context": {"machine_family": "PRESSA_TEST", "layout_id": "TEST_LAYOUT"}},
            }
        ),
        encoding="utf-8",
    )
    (export_dir / "quality_gate_summary.json").write_text(
        json.dumps(
            {
                "counts": {"components_total": 11, "components_training_ready": 9},
                "outputs": {"training_ready_count": 9},
                "gate": {"status": "pass", "passed": True},
            }
        ),
        encoding="utf-8",
    )

    args = types.SimpleNamespace(
        shared_root=str(tmp_path / "shared"),
        session_id=None,
        mold_id=None,
        material_id=None,
        machine_family=None,
        limit=10,
        json=True,
    )

    rc = _handle_predictive_list_artifacts(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["session_id"] == "session-123"
    assert payload[0]["mold_id"] == "stampo_giotto"
    assert payload[0]["material_id"] == "pc"
    assert payload[0]["machine_family"] == "PRESSA_TEST"
    assert payload[0]["rows_total"] == 11
    assert payload[0]["rows_training_ready"] == 9


@patch("moldvision.publish.publish_bundle")
@patch("moldvision.predictive.suggestion_bundle.write_suggestion_bundle")
def test_predictive_bundle_can_publish_directly(mock_write_bundle, mock_publish, tmp_path, capsys) -> None:
    from moldvision.cli_handlers import _handle_predictive_bundle

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    with (train_dir / "train_result.pkl").open("wb") as fh:
        pickle.dump({"ok": True}, fh)
    (train_dir / "scope.json").write_text(
        json.dumps({"mold_id": "stampo_giotto", "material_id": "pc", "machine_id": "PRESSA_TEST"}),
        encoding="utf-8",
    )

    bundle_dir = train_dir / "deploy" / "suggestions-v1.0.0"
    bundle_dir.mkdir(parents=True)
    mock_write_bundle.return_value = bundle_dir
    mock_publish.return_value = {"bundle_id": "suggestions-v1.0.0", "publish_target": "shared"}

    args = types.SimpleNamespace(
        train_dir=str(train_dir),
        model_name="startup-suggestion",
        model_version="1.0.0",
        channel="stable",
        supersedes=None,
        sugbundle=False,
        mold_id=None,
        material_id=None,
        machine_id=None,
        publish=True,
        publish_dry_run=False,
    )

    rc = _handle_predictive_bundle(args)

    assert rc == 0
    mock_publish.assert_called_once_with(
        bundle_dir,
        role="startup_suggestion",
        channel="stable",
        dry_run=False,
    )
    assert "Published:" in capsys.readouterr().out


@patch("moldvision.predictive.trainer.train_suggestion_models")
@patch("moldvision.predictive.training_row_loader.assess_training_readiness")
@patch("moldvision.predictive.training_row_loader.check_schema_homogeneity")
@patch("moldvision.predictive.training_row_loader.summarize_scope_distribution")
@patch("moldvision.predictive.training_row_loader.validate_dataset")
@patch("moldvision.predictive.training_row_loader.load_training_rows")
def test_predictive_train_defaults_to_local_predictive_runs_root(
    mock_load_rows,
    mock_validate,
    mock_scope_distribution,
    mock_homogeneity,
    mock_readiness,
    mock_train,
    tmp_path,
    capsys,
) -> None:
    from moldvision.cli_handlers import _handle_predictive_train

    input_path = tmp_path / "training_rows.jsonl"
    input_path.write_text("{}", encoding="utf-8")
    local_runs_root = tmp_path / "local-runs"

    mock_load_rows.return_value = [
        {
            "mold_id": "stampo_giotto",
            "material_id": "pc",
            "context": {"machine_family": "PRESSA_TEST"},
            "eligibility": {"training_ready": True},
        }
    ]
    mock_validate.return_value = {"valid": True, "invalid_rows": 0, "row_errors": []}
    mock_scope_distribution.return_value = {"distinct_scope_count": 1}
    mock_homogeneity.return_value = {"homogeneous": True}
    mock_readiness.return_value = {"level": "good", "message": "ready"}
    mock_train.return_value = types.SimpleNamespace(
        targets={
            "quality_score": types.SimpleNamespace(
                cv_metric_name="rmse",
                cv_metric_value=0.1234,
                cv_metric_std=0.01,
            )
        },
        n_eligible_rows=1,
        feature_keys=["pressure_injection:step_1.setpoint"],
    )

    args = types.SimpleNamespace(
        input=str(input_path),
        output_dir=None,
        cv_folds=5,
        n_estimators=300,
        learning_rate=0.05,
        null_strategy="native_missing",
        min_feature_presence_ratio=0.05,
        mold_id=None,
        material_id=None,
        machine_id=None,
        allow_scope_filtering=False,
    )

    with patch("moldvision.cli_handlers.appconfig.get_predictive_runs_root", return_value=local_runs_root):
        rc = _handle_predictive_train(args)

    assert rc == 0
    run_dirs = [p for p in local_runs_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "train_result.pkl").exists()
    assert (run_dir / "scope.json").exists()
    scope = json.loads((run_dir / "scope.json").read_text(encoding="utf-8"))
    assert scope == {
        "mold_id": "stampo_giotto",
        "material_id": "pc",
        "machine_id": "PRESSA_TEST",
    }
    output = capsys.readouterr().out
    assert "Output directory not provided; using local predictive run folder:" in output
