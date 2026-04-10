# Bundle Contract — ARIA MoldVision ↔ ARIA MoldPilot

This document defines the deployment bundle format that MoldVision produces and
MoldPilot consumes. Both projects must adhere to this contract to ensure bundles
install and run correctly.

---

## Bundle Directory Layout

A bundle is a directory (or a `.mpk` / `.zip` archive of that directory) with
the following files:

```
<bundle_dir>/
├── manifest.json          # ← required by MoldPilot (see schema below)
├── model.onnx             # primary inference artifact (float32)
├── model_fp16.onnx        # optional — FP16 variant (GPU only)
├── model_quantized.onnx   # optional — INT8 quantized variant
├── model.engine           # optional — TensorRT engine
├── preprocess.json        # preprocessing contract
├── postprocess.json       # postprocessing defaults
├── classes.json           # class list (legacy; classes are inlined into manifest.json)
├── model_config.json      # training provenance (task, size, resolution, …)
├── checkpoint.pth         # PyTorch weights (debug fallback, optional)
│
│  — standalone mode only (--standalone flag) —
├── infer.py               # standalone Python runner
├── requirements.txt       # ONNX runtime requirements
├── requirements-pytorch-fallback.txt
└── moldvision/            # vendored package for standalone runner
```

MoldPilot only reads `manifest.json`, `preprocess.json`, `postprocess.json`, and
the ONNX model file(s). All other files are ignored by MoldPilot but may be
useful for debugging or standalone usage.

---

## `manifest.json` Schema

MoldPilot's `LocalModelRegistryService` and `OnnxInferenceService` both read
`manifest.json`. The following fields are **required**:

| Field | Type | Description |
|---|---|---|
| `bundle_id` | string | Unique identifier. Convention: `{model-name}-v{version}` e.g. `surface-defect-detector-v1.2.0` |
| `model_name` | string | Human-readable name shown in MoldPilot UI |
| `model_version` | string | Semantic version string, e.g. `1.2.0` |
| `channel` | `"stable"` \| `"beta"` | Release channel. `beta` bundles are installed but not auto-activated |
| `supersedes` | string \| null | `bundle_id` of the previous version this release replaces |
| `min_app_version` | string | Minimum MoldPilot version required, e.g. `"0.3.0"`. Use `"0.0.0"` for no restriction |
| `classes` | object | Class map: `{"0": "label0", "1": "label1", …}` — string-keyed integer IDs |
| `runtime.providers` | array | Ordered ONNX Runtime provider list, e.g. `["CUDAExecutionProvider", "CPUExecutionProvider"]` |
| `checksums` | object | SHA-256 hex digests keyed by filename, e.g. `{"model.onnx": "abcdef…"}`. Covers all files except `manifest.json` itself |

The following fields are written by MoldVision for provenance but are optional
from MoldPilot's perspective:

`format_version`, `created_at`, `dataset_dir`, `source_weights`,
`primary_artifact`, `artifacts`, `tensorrt`, `checkpoint`, `runtime_versions`,
`standalone`.

### Minimal valid `manifest.json`

```json
{
  "bundle_id": "surface-defect-detector-v1.0.0",
  "model_name": "Surface Defect Detector",
  "model_version": "1.0.0",
  "channel": "stable",
  "supersedes": null,
  "min_app_version": "0.0.0",
  "classes": {
    "0": "scratch",
    "1": "dent",
    "2": "stain"
  },
  "runtime": {
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
  },
  "checksums": {
    "model.onnx": "a3f1...",
    "preprocess.json": "b9c2...",
    "postprocess.json": "d7e4..."
  }
}
```

---

## `preprocess.json` Contract

MoldPilot's `OnnxInferenceService` reads these fields at inference time:

| Field | Expected value |
|---|---|
| `resize_policy` | `"letterbox"` — aspect-ratio preserving with grey (114,114,114) padding |
| `target_h` / `target_w` | Integer pixel dimensions, e.g. `640` |
| `input_color` | `"RGB"` |
| `input_layout` | `"NCHW"` |
| `input_dtype` | `"float32"` (or `"float16"` for FP16 model) |
| `input_range` | `"0..1"` |
| `normalize.mean` | `[0.485, 0.456, 0.406]` (ImageNet) |
| `normalize.std` | `[0.229, 0.224, 0.225]` (ImageNet) |

MoldVision writes these values automatically and they match MoldPilot's
`OnnxInferenceService._preprocess()` implementation exactly.

---

## `postprocess.json` Contract

MoldPilot reads these fields to apply post-processing thresholds:

| Field | Description |
|---|---|
| `score_threshold_default` | Confidence cut-off, e.g. `0.5` |
| `nms_iou_threshold_default` | IoU threshold for NMS, e.g. `0.3` |
| `topk_default` | Max queries to evaluate, e.g. `100` |
| `max_dets_default` | Max detections to return per frame |

---

## Creating a Bundle (MoldVision)

```powershell
# Minimal — produces a MoldPilot-ready bundle
moldvision bundle `
  --dataset-dir datasets/<UUID> `
  --weights datasets/<UUID>/models/checkpoint_portable.pth `
  --model-name "Surface Defect Detector" `
  --model-version 1.0.0 `
  --mpk

# Full version bump — supersedes a previous bundle
moldvision bundle `
  --dataset-dir datasets/<UUID> `
  --weights datasets/<UUID>/models/checkpoint_portable.pth `
  --model-name "Surface Defect Detector" `
  --model-version 1.1.0 `
  --supersedes surface-defect-detector-v1.0.0 `
  --channel stable `
  --mpk

# Beta / staged rollout
moldvision bundle `
  --dataset-dir datasets/<UUID> `
  --weights datasets/<UUID>/models/checkpoint_portable.pth `
  --model-version 2.0.0-beta.1 `
  --channel beta `
  --mpk

# Standalone bundle (for use without MoldPilot)
moldvision bundle `
  --dataset-dir datasets/<UUID> `
  --weights datasets/<UUID>/models/checkpoint_portable.pth `
  --model-version 1.0.0 `
  --standalone `
  --zip
```

The `--mpk` flag writes `<bundle_dir>.mpk` alongside the bundle directory.
MoldPilot's `install_bundle()` and `install_update()` accept `.mpk` directly.

---

## Installing a Bundle (MoldPilot)

MoldPilot's `LocalModelRegistryService` handles installation:

```python
from pathlib import Path
from aria_moldpilot.infrastructure.model_registry import LocalModelRegistryService

registry = LocalModelRegistryService(models_dir=Path("models"))

# First install
ref = registry.install_bundle(Path("surface-defect-detector-v1.0.0.mpk"))
registry.set_active_bundle(ref.bundle_id)

# Version update (marks the superseded bundle as inactive, activates new one)
ref = registry.install_update(Path("surface-defect-detector-v1.1.0.mpk"))
```

### Integrity verification

```python
ok = registry.verify_bundle("surface-defect-detector-v1.1.0")
# Recomputes SHA-256 of every file listed in manifest.json["checksums"] and
# compares against stored values. Returns False on any mismatch.
```

---

## Remote Bundle Updates (Future Work)

MoldPilot currently has **no remote download capability** — it installs from
local `.mpk` / `.zip` files only. To enable over-the-air model updates from a
server, the following additions are needed:

### Recommended server contract

Host a simple JSON index file at a stable URL, e.g.:
`https://your-server/models/index.json`

```json
{
  "schema": "model-index-v1",
  "updated_at": "2026-03-29T12:00:00Z",
  "bundles": [
    {
      "bundle_id": "surface-defect-detector-v1.1.0",
      "model_name": "Surface Defect Detector",
      "model_version": "1.1.0",
      "channel": "stable",
      "supersedes": "surface-defect-detector-v1.0.0",
      "url": "https://your-server/models/surface-defect-detector-v1.1.0.mpk",
      "sha256": "abc123..."
    }
  ]
}
```

### Recommended MoldPilot additions

Add a `RemoteModelRegistryService` (or extend `LocalModelRegistryService`) that:

1. `check_for_update(index_url: str, channel: str = "stable") -> BundleRef | None`  
   — Fetches `index_url`, compares available `model_version` vs active bundle
     version using `packaging.version.Version` comparison. Returns the latest
     bundle metadata if an update is available, `None` otherwise.

2. `download_and_install(bundle_url: str, expected_sha256: str, *, activate: bool = True) -> ModelBundleRef`  
   — Downloads the `.mpk` to a temp file, verifies SHA-256, then calls
     `install_update()`.

The server just needs to serve static files — no special backend required.
Upload new `.mpk` files and update `index.json` after each MoldVision training
run.

### MoldVision workflow

```powershell
# 1. Train a new model
moldvision train --dataset-dir datasets/<UUID> --epochs 100 ...

# 2. Create a MoldPilot-ready bundle with the new version
moldvision bundle `
  --dataset-dir datasets/<UUID> `
  --weights datasets/<UUID>/models/checkpoint_portable.pth `
  --model-version 1.2.0 `
  --supersedes surface-defect-detector-v1.1.0 `
  --mpk

# 3. Upload <bundle>.mpk to the server
# 4. Update index.json on the server
# 5. MoldPilot next startup check detects update and downloads automatically
```

---

## Startup-Suggestion Bundle (`.sugbundle`)

A separate bundle format for the Startup Assistant's GBT (LightGBM → ONNX)
models.  These bundles are produced by `moldvision predictive bundle` and
installed into MoldPilot via `ModelRegistryService.install_suggestion_bundle()`.

### Directory layout

```
<bundle_dir>/
├── manifest.json          # metadata + checksums (schema below)
├── model_quality_score.onnx        # regression model — quality score 0–1
├── model_defect_burn_mark.onnx     # binary classifier — burn mark probability
├── model_defect_flash.onnx         # binary classifier — flash probability
├── model_defect_sink_mark.onnx     # binary classifier — sink mark probability
├── model_defect_weld_line.onnx     # binary classifier — weld line probability
└── training_meta.json     # training provenance
```

Pack with `moldvision predictive bundle --pack` to produce `<name>.sugbundle`
(a plain ZIP archive of the directory above).

### `manifest.json` schema

| Field | Type | Description |
|---|---|---|
| `bundle_type` | `"startup_suggestion"` | Discriminates from CV inference bundles |
| `bundle_id` | string | `{model-name}-suggest-v{version}` |
| `model_name` | string | Human-readable label shown in MoldPilot UI |
| `model_version` | string | Semantic version string |
| `format_version` | `1` | Schema version (int) |
| `created_at` | ISO 8601 string | UTC timestamp when bundle was written |
| `feature_keys` | string[] | Ordered list of MoldTrace feature keys expected at inference |
| `imputation` | object | Mean imputation values: `{"<feature_key>": <float>, …}` — applied before inference when a feature is missing |
| `target_models` | object | Map of target name → ONNX filename: `{"quality_score": "model_quality_score.onnx", …}` |
| `quality_weights` | object | Defect-to-quality weight map used by MoldPilot Tier 1 local search: `{"burn_mark": 0.20, "flash": 0.30, "sink_mark": 0.35, "weld_line": 0.15}` |
| `checksums` | object | SHA-256 hex per ONNX file and `training_meta.json` |

### `training_meta.json` schema

```json
{
  "n_rows": 1240,
  "cv_metrics": {
    "quality_score": {"rmse": 0.082},
    "defect_burn_mark": {"auc": 0.91},
    "defect_flash": {"auc": 0.88},
    "defect_sink_mark": {"auc": 0.93},
    "defect_weld_line": {"auc": 0.85}
  },
  "date": "2026-04-10T12:00:00"
}
```

### ONNX model conventions

All five ONNX models follow the same contract:

- **Input**: `float_input` — shape `[1, n_features]`, `float32`, NaN-safe
- **Regression** (`quality_score`) — single output `variable` shape `[1, 1]`
- **Classification** (`defect_*`) — two outputs:
  - `label` — predicted class (unused by MoldPilot)
  - `probabilities` — shape `[1, 2]`; MoldPilot reads `probabilities[0][1]` (positive-class probability)
- Exported with `onnxmltools.convert_lightgbm(..., zipmap=False)` — no ZipMap post-processing

### Creating a `.sugbundle` (MoldVision)

```powershell
# 1. Export training rows from ARIA MoldTrace (JSONL, training_row_v1 schema)
#    Place in e.g. C:\data\training_rows.jsonl

# 2. Train the five GBT models
moldvision predictive train `
  --input C:\data\training_rows.jsonl `
  --output-dir runs\suggest-v1 `
  --model-name "Mold A Startup Suggestion" `
  --model-version 1.0.0

# 3. Pack the suggestion bundle
moldvision predictive bundle `
  --train-dir runs\suggest-v1 `
  --output-dir bundles\suggest-v1 `
  --pack
# → bundles\suggest-v1\mold-a-startup-suggestion-v1.0.0.sugbundle
```

### Installing a `.sugbundle` (MoldPilot)

```python
from pathlib import Path
from aria_moldpilot.infrastructure.model_registry import LocalModelRegistryService

registry = LocalModelRegistryService(models_dir=Path("models"))
ref = registry.install_suggestion_bundle(
    Path("mold-a-startup-suggestion-v1.0.0.sugbundle")
)
# MoldPilot app.py auto-loads the active suggestion bundle at startup
# and upgrades the Startup Assistant from Tier 0 (rule-based) to Tier 1 (GBT).
```

Suggestion bundles are tracked in `registry_suggestions.json` alongside the
main `registry.json` and stored under `models/suggestion_bundles/`.


---

## Versioning Convention

Bundle IDs follow the pattern `{model-name}-v{major}.{minor}.{patch}`.

- Bump **patch** for retraining on new data with the same architecture.
- Bump **minor** for new classes or preprocessing changes.
- Bump **major** for architecture changes (new model size, task, input resolution).

When bumping version, always set `--supersedes` to the previous `bundle_id` so
MoldPilot can mark the old bundle as superseded and enable rollback.
