# MoldVision Predictive Model Plan
## Startup-Suggestion GBT Pipeline

This document describes how to produce and deploy the LightGBM models that
power MoldPilot's Tier 1 Startup Assistant.

---

## Overview

The Startup Assistant suggests injection-moulding parameter adjustments to
prevent defects at shot startup.  It operates in two tiers:

| Tier | Engine | Requires |
|------|--------|----------|
| 0 | `RuleBasedStartupAssistantService` — static rule table | Nothing extra |
| 1 | `Tier1StartupAssistantService` — GBT local search over ONNX models | `.sugbundle` installed in MoldPilot |

MoldPilot auto-detects the best available tier at startup.  Tier 1 activates
as soon as a valid `.sugbundle` is installed.

---

## Training Data Contract

Models are trained on **`training_row_v1` JSONL** records produced by
**ARIA MoldTrace**.  Each record documents one production session and must
contain:

```json
{
  "schema": "training_row_v1",
  "session_id": "...",
  "mold_id": "...",
  "quality_score": 0.82,
  "defect_flags": {
    "burn_mark": false,
    "flash": true,
    "sink_mark": false,
    "weld_line": false
  },
  "parameters": {
    "injection_speed:slot_0.mean": 42.5,
    "mold_temp:slot_0.mean": 58.1,
    "holding_pressure:slot_0.mean": 650.0
  }
}
```

**Required fields per record:**

| Field | Type | Notes |
|---|---|---|
| `quality_score` | float 0–1 | Primary regression target |
| `defect_flags.burn_mark` | bool | Classification target |
| `defect_flags.flash` | bool | Classification target |
| `defect_flags.sink_mark` | bool | Classification target |
| `defect_flags.weld_line` | bool | Classification target |
| `parameters.*` | float | Any MoldTrace parameter features |

Missing parameter values are allowed — LightGBM handles `NaN` natively.
The union schema across all rows determines the feature vector; mean imputation
values are stored in the bundle for runtime use.

**Minimum recommended dataset:** 300+ rows spanning at least 3 different
quality levels.  With fewer rows the GBT models will overfit and CV metrics
will be unreliable.

---

## Five Training Targets

| Target | Type | Metric | Weight in local search |
|---|---|---|---|
| `quality_score` | Regression (LGBMRegressor) | RMSE | Primary objective |
| `defect_burn_mark` | Binary (LGBMClassifier) | AUC | 0.20 |
| `defect_flash` | Binary (LGBMClassifier) | AUC | 0.30 |
| `defect_sink_mark` | Binary (LGBMClassifier) | AUC | 0.35 |
| `defect_weld_line` | Binary (LGBMClassifier) | AUC | 0.15 |

Weights (§8.3.1 of ARIA_System_Integration.md) reflect the relative impact of
each defect class on part rejection rate in the reference production environment.
Adjust them in `GbtTrainingConfig.quality_weights` if your rejection profile
differs.

---

## Training Recipe

### 1. Install optional dependencies

```powershell
pip install "aria-moldvision[predictive]"
# Installs: lightgbm>=4.3, onnxmltools>=1.12, skl2onnx>=1.17, scikit-learn>=1.4
```

### 2. Export training rows from MoldTrace

Use ARIA MoldTrace's export tool to produce a JSONL file of `training_row_v1`
records.  Place the file somewhere accessible, e.g.:

```
C:\data\mold_a_training_rows.jsonl
```

Validate the data before training:

```powershell
moldvision predictive validate-dataset --input C:\data\mold_a_training_rows.jsonl
```

This reports:
- Row count and schema conformance
- Fraction of rows with each defect flag
- Missing-value rates per feature column
- Class imbalance warnings

### 3. Train the models

```powershell
moldvision predictive train `
  --input C:\data\mold_a_training_rows.jsonl `
  --output-dir runs\mold-a-v1 `
  --model-name "Mold A Startup Suggestion" `
  --model-version 1.0.0
```

Outputs written to `runs\mold-a-v1\`:

| File | Contents |
|---|---|
| `train_result.pkl` | Full `TrainResult` object (needed by `predictive bundle`) |
| `training_meta.json` | Row count + 5-fold CV metrics |
| `model_quality_score.onnx` | Regression ONNX model |
| `model_defect_burn_mark.onnx` | Classification ONNX model |
| `model_defect_flash.onnx` | Classification ONNX model |
| `model_defect_sink_mark.onnx` | Classification ONNX model |
| `model_defect_weld_line.onnx` | Classification ONNX model |

**Acceptable CV metrics (minimum thresholds before deploying):**

| Target | Threshold |
|---|---|
| `quality_score` RMSE | ≤ 0.15 |
| `defect_*` AUC | ≥ 0.75 |

### 4. Pack the suggestion bundle

```powershell
moldvision predictive bundle `
  --train-dir runs\mold-a-v1 `
  --output-dir bundles\mold-a-v1 `
  --pack
```

Produces `bundles\mold-a-v1\mold-a-startup-suggestion-v1.0.0.sugbundle`.

### 5. Install in MoldPilot

```powershell
# Copy or move the .sugbundle to the machine running MoldPilot
# Then use the Python API or (future) MoldPilot UI to install:

python - << 'EOF'
from pathlib import Path
from aria_moldpilot.infrastructure.model_registry import LocalModelRegistryService
import os

models_dir = Path(os.environ["LOCALAPPDATA"]) / "ARIA" / "MoldPilot" / "models"
registry = LocalModelRegistryService(models_dir=models_dir)
ref = registry.install_suggestion_bundle(
    Path(r"bundles\mold-a-v1\mold-a-startup-suggestion-v1.0.0.sugbundle")
)
print(f"Installed: {ref}")
EOF
```

Restart MoldPilot.  The log line on startup will change from:

```
Startup Assistant: no suggestion bundle installed, using Tier 0 (rule-based).
```

to:

```
Startup Assistant: loaded suggestion bundle mold-a-startup-suggestion-v1.0.0 → Tier 1 active.
```

---

## Tier 1 Inference in MoldPilot

When a suggestion bundle is installed, `Tier1StartupAssistantService`:

1. Reads current HMI parameter values from the operator's active configuration.
2. Constructs a `float32` feature vector using the bundle's `feature_keys` list
   (NaN for any missing parameter).
3. Runs all 5 ONNX models: gets `quality_score` + 4 defect probabilities.
4. Performs a **local search**: perturbs each parameter ±5 % and ±10 %, re-runs
   inference, computes Δ`quality_score`.
5. Returns the top-N parameter adjustments with the largest positive Δ quality as
   `StartupSuggestion` objects, ranked by impact.

The workspace widget receives metric values via `push_metric()` calls from the
monitoring controller and passes them to the assistant, which returns actionable
suggestions shown in the right panel.

---

## Retraining

Re-run Steps 2–5 whenever:

- A new mold or material is introduced.
- Defect rates shift significantly from the training distribution.
- At least 50 new labelled production sessions are available.

There is no automated retraining trigger yet.  Track training runs by
incrementing the `--model-version` flag and store the resulting
`.sugbundle` alongside the CV metric report.

---

## File Locations

| Artefact | Location |
|---|---|
| Training JSONL | `C:\data\<mold_id>_training_rows.jsonl` (operator-managed) |
| Training outputs | `runs\<name>-v<version>\` (gitignored) |
| Packed bundle | `bundles\<name>-v<version>\*.sugbundle` (gitignored) |
| Installed bundle | `%LOCALAPPDATA%\ARIA\MoldPilot\models\suggestion_bundles\` |
| Bundle registry | `%LOCALAPPDATA%\ARIA\MoldPilot\models\registry_suggestions.json` |

---

## See Also

- `docs/BUNDLE_CONTRACT.md` — `.sugbundle` format specification
- `docs/ARIA_System_Integration.md` — §8.3 Startup Assistant, feature schema
- `moldvision/predictive/trainer.py` — `GbtTrainingConfig`, `train_suggestion_models()`
- `moldvision/predictive/suggestion_bundle.py` — `write_suggestion_bundle()`, `pack_sugbundle()`
- `ARIA_MoldPilot/src/aria_moldpilot/infrastructure/tier1_startup_assistant.py` — Tier 1 engine
