# ARIA_MoldVision

Integrated system for defect detection models, from dataset preparation to training and deployment.

## What you can do

- Create and manage dataset folders (UUID layout)
- List all datasets and inspect their split counts and training status
- Extract frames from videos for labeling (by count or by target fps)
- Convert YOLO → COCO (detect or seg)
- Validate COCO splits before training
- Subsample COCO splits (stratified by class + proportional background)
- Normalize COCO category ids to contiguous `0..N-1` (fixes common training issues)
- Train defect detection models (detect or seg)
- Export trained models to ONNX (and optionally build TensorRT engines)
- Create deployment bundles compatible with ARIA MoldPilot (`.mpk` format, versioned, checksummed)
- Pre-label new images with a trained model via Label Studio (active learning loop)
- Store persistent defaults (dataset root, num-workers, backend, export format)
- Run `doctor` to check your environment and print common fix hints

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Install PyTorch separately to match your CUDA/CPU setup, then:
pip install -r requirements.txt
pip install -e .
```

After `pip install -e .` the `moldvision` command is available in your venv.
Use `moldvision --help` or `python -m moldvision --help` interchangeably.

## Configuration

MoldVision stores persistent defaults in `%LOCALAPPDATA%\MoldVision\config.json` (Windows) or the appropriate platform config directory.

```powershell
moldvision config show                          # print all current settings
moldvision config set dataset-root D:\datasets  # default root for dataset create/list
moldvision config set num-workers 4             # dataloader workers (default: 0 on Windows)
moldvision config set inference-backend onnx    # default backend for moldvision infer
moldvision config set export-format onnx_fp16   # default format for moldvision export
```

Settings are overridden by explicit CLI flags and by environment variables (`MOLDVISION_DATASET_ROOT`, `MOLDVISION_NUM_WORKERS`, `MOLDVISION_BACKEND`, `MOLDVISION_EXPORT_FORMAT`).

> **Windows tip:** `num-workers` defaults to `0` to avoid DataLoader multiprocessing issues. Set it once with `config set` instead of passing `--num-workers 0` on every training run.

## Typical workflow

### 1) Create a dataset folder

```powershell
moldvision dataset create --name "my-dataset" -c monitor -c keyboard
moldvision dataset create --name "my-dataset" --classes-file classes.txt  # one class per line
```

### 1b) (Optional) Extract frames from videos for labeling

Extract frames uniformly across one or more video files:

```powershell
# Explicit file list — extract 500 frames total
moldvision dataset extract-frames -v video1.mp4 -v video2.mp4 -n 500 -d datasets/<UUID>

# Scan a folder for all videos
moldvision dataset extract-frames --videos-dir D:\recordings -n 500 -d datasets/<UUID>

# Target frame rate instead of a fixed count (e.g. 1 frame every 2 seconds = 0.5 fps)
moldvision dataset extract-frames --videos-dir D:\recordings --fps 0.5 -d datasets/<UUID>

# Filter by extension when scanning a folder (default: mp4,avi,mov,mkv,webm)
moldvision dataset extract-frames --videos-dir D:\recordings --ext mp4,mov --fps 1 -d datasets/<UUID>

# Custom output directory (overrides --dataset-dir)
moldvision dataset extract-frames -v video1.mp4 -n 100 -o my_frames/
```

Frames are saved in `datasets/<UUID>/raw/`. You can mix `--videos` and `--videos-dir` in the same command.

### 1c) (Optional) Inspect your datasets

```powershell
moldvision dataset list                         # table of all datasets under configured root
moldvision dataset list --root D:\other         # list a specific root
moldvision dataset info -d datasets/<UUID>      # split counts, class names, model status, exports
```

### 2) Put images in `datasets/<UUID>/raw/`

If you have **mixed labels** (some in YOLO, some already in COCO):
- Put YOLO `*.txt` files in `labels_inbox/yolo/`
- Put COCO `_annotations.coco.json` files in `labels_inbox/coco/` (images should already be in `raw/`)
- Run one command to ingest everything into `coco/train` + `coco/valid`:

```powershell
moldvision dataset ingest -d datasets/<UUID> --train-ratio 0.8 --seed 0
```

The ingest step resolves classes from `METADATA.json`, splits by ratio, **includes unlabeled images from `raw/` as background** (empty annotations), and quarantines conflicting labels into `labels_inbox/quarantine/`.

Important:
- If your YOLO labels are polygons (segmentation), add `--yolo-task seg`.
- If you accidentally ingested with the wrong task and want a clean restart:

```powershell
moldvision dataset reset-coco -d datasets/<UUID>
# then re-run dataset ingest
```

**Merge an external COCO export:**

```powershell
moldvision dataset import-coco -d datasets/<UUID> --split train `
  --coco-json path\to\_annotations.coco.json --images-dir path\to\images `
  --mode copy --align-metadata
```

### 3) Convert YOLO → COCO (if not using `ingest`)

```powershell
moldvision dataset yolo-to-coco -d datasets/<UUID> --task seg --train-ratio 0.8 --copy-images --validate
```

### 4) Validate and clean COCO splits

```powershell
moldvision dataset validate -d datasets/<UUID> --task seg
```

If you see warnings about 1-indexed category ids or holes, normalize ids (creates `.bak` backups):

```powershell
moldvision dataset normalize-coco-ids -d datasets/<UUID>
```

If you see duplicated class names or wrong `num_classes`, align COCO categories to `METADATA.json`:

```powershell
moldvision dataset align-metadata -d datasets/<UUID>
```

**Subsample a split** (guarantees at least one instance per class, preserves background image ratio):

```powershell
# Single split — keep 20 % of train images
moldvision dataset subsample -d datasets/<UUID> --split train --fraction 0.2

# All splits at once (skips any that do not exist)
moldvision dataset subsample -d datasets/<UUID> --split all --max-images 500

# Dry-run to preview without writing
moldvision dataset subsample -d datasets/<UUID> --split train --fraction 0.2 --dry-run
```

The command prints a per-class instance/image count table **before and after** subsampling so you can verify the result.

### 5) Train

```powershell
moldvision train -d datasets/<UUID> --task seg --epochs 20 --batch-size 4 --grad-accum 4 --lr 1e-4
```

Common training options:
- `--num-workers` defaults to the value set in `moldvision config set num-workers` (0 on Windows). Override per-run with `--num-workers N`.
- Training fine-tunes from the selected model size's default pretrained weights by default.
- To train from scratch: `--no-pretrained`
- To override the pretrained source: `--pretrain-weights path\to\weights.pth`
- Start a new run from existing weights: `--finetune-from path\to\checkpoint.pth`
- Continue an existing run: `--resume path\to\checkpoint.pth`
- Evaluation only (no training): `--eval-only`

## Troubleshooting

```powershell
moldvision doctor
```

## Export (deployment)

The default export format is set by `moldvision config set export-format <fmt>`. Override per-run with `--format`.

```powershell
# ONNX (auto-detects task and size from model_config.json)
moldvision export -d datasets/<UUID> -w datasets/<UUID>/models/checkpoint_best_total.pth --format onnx

# FP16 — ~2x speedup on modern GPUs, no calibration needed
moldvision export -d datasets/<UUID> -w ... --format onnx_fp16

# INT8 quantization — static calibration from your dataset
moldvision export -d datasets/<UUID> -w ... --format onnx_quantized
#   --calibration-split valid   (default: valid)
#   --calibration-count 100     (default: 100)

# TensorRT engine (requires trtexec on PATH)
moldvision export -d datasets/<UUID> -w ... --format tensorrt --fp16
```

> Tip: export is `--strict` by default (fails fast on mismatched checkpoints). Use `--non-strict` only for debugging.

## Deployment Bundle (ARIA MoldPilot compatible)

Creates a folder containing: `manifest.json` (versioned, with SHA-256 checksums),
`model.onnx`, `preprocess.json`, `postprocess.json`, `classes.json`, `model_config.json`,
and `checkpoint.pth`. The bundle installs directly into ARIA MoldPilot via `install_bundle()`.

```powershell
# Recommended — MoldPilot-ready bundle with .mpk archive
moldvision bundle `
  -d datasets/<UUID> `
  -w datasets/<UUID>/models/checkpoint_portable.pth `
  --model-name "Surface Defect Detector" `
  --model-version 1.0.0 `
  --mpk

# Version bump — supersedes a previous bundle
moldvision bundle `
  -d datasets/<UUID> -w ... `
  --model-version 1.1.0 `
  --supersedes surface-defect-detector-v1.0.0 `
  --mpk

# Beta / staged rollout
moldvision bundle -d datasets/<UUID> -w ... --model-version 2.0.0 --channel beta --mpk

# Include INT8 quantized model
moldvision bundle -d datasets/<UUID> -w ... --export onnx_quantized --mpk

# Include FP16 model (GPU inference)
moldvision bundle -d datasets/<UUID> -w ... --export onnx_fp16 --mpk

# Include TensorRT engine (requires trtexec on PATH)
moldvision bundle -d datasets/<UUID> -w ... --export tensorrt --mpk

# Standalone bundle with Python runner (for use without MoldPilot)
moldvision bundle -d datasets/<UUID> -w ... --standalone --zip
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--model-name` | dataset name | Human-readable name shown in MoldPilot |
| `--model-version` | `1.0.0` | Semantic version string |
| `--channel` | `stable` | `stable` or `beta` |
| `--supersedes` | — | `bundle_id` of the version this replaces |
| `--min-app-version` | `0.0.0` | Minimum MoldPilot version required |
| `--mpk` | off | Write `<bundle>.mpk` (MoldPilot install format) |
| `--zip` | off | Write `<bundle>.zip` |
| `--standalone` | off | Add `infer.py`, `requirements.txt`, vendored package |

The `bundle_id` is auto-generated as `{model-name}-v{version}` if not provided via `--bundle-id`.

> See `docs/BUNDLE_CONTRACT.md` for the full manifest schema, MoldPilot install API,
> and the remote update design (server-side index.json contract).

### Smoke-test a bundle with `moldvision infer`

```powershell
moldvision infer --bundle-dir datasets/<UUID>/deploy/<bundle> --image path\to\test.jpg
```

## Pre-labeling with Label Studio (active learning)

Use a trained MoldVision model to pre-annotate new images in Label Studio.
Annotators see model predictions as a starting point and only need to correct
mistakes — typically 60–80 % faster than labeling from scratch.

**Install:**
```powershell
pip install label-studio
pip install "aria-moldvision[label-studio]"
```

**Start the pre-labeling backend:**
```powershell
$env:MOLDVISION_BUNDLE_DIR = "datasets/<UUID>/deploy/<bundle>"
label-studio-ml start moldvision\label_studio_backend.py --port 9090
```

**Then in Label Studio:**
1. Create a project with the appropriate label config (bounding boxes for detect, add `PolygonLabels` for seg).
2. **Settings → Machine Learning → Add Model** → `http://localhost:9090`.
3. Import new unlabeled images — pre-annotations appear automatically.
4. Review, correct, export COCO JSON, ingest into your dataset, retrain.

The backend supports both detect (`rectanglelabels`) and seg (`polygonlabels`) bundles
and auto-discovers tag names from your project's label config — no hardcoding needed.

Override the confidence threshold at start time: `--with score_threshold=0.7`

> See `docs/LABELING_WORKFLOW.md` for the complete step-by-step guide including
> COCO export, dataset ingest, and the full active learning loop.

## Tools (optional scripts)

The `scripts/` folder contains standalone utilities for dataset prep and inference — not required for the main training workflow:

| Script | Purpose |
|---|---|
| `scripts/infer_image.py` | Run inference on a single image; draw bounding boxes and masks |
| `scripts/infer_video.py` | Run inference on a video file and optionally save an annotated output |
| `scripts/infer_webcam.py` | Live inference from a webcam with real-time overlay |
| `scripts/batch_infer.py` | Run inference over all images in a COCO split; outputs a COCO-like JSON |
| `scripts/visualize_annotations.py` | Sample images from COCO splits and save annotated visualizations |
| `scripts/remove_coco_class.py` | Remove one or more categories from COCO JSON files (backs up originals) |

**Video inference examples:**

```powershell
python scripts/infer_video.py --video path\to\input.mp4 `
  --weights datasets/<UUID>/models/checkpoint_best_total.pth `
  --task detect --output-video runs\video_detect.mp4

python scripts/infer_video.py --video path\to\input.mp4 `
  --weights datasets/<UUID>/models/checkpoint_best_total.pth `
  --task seg --classes-file datasets/<UUID>/METADATA.json --out-dir runs\videos
```

See `docs/TOOLS.md` for notes and recommendations.

## Documentation

| File | Contents |
|---|---|
| `docs/TOOLS.md` | Notes on optional scripts and inference utilities |
| `docs/TRANSFER_AND_INFERENCE.md` | Model transfer and inference workflow notes |
| `docs/BUNDLE_CONTRACT.md` | Bundle format spec, `manifest.json` schema, MoldPilot integration, remote update design |
| `docs/LABELING_WORKFLOW.md` | Full Label Studio active-learning loop: pre-label → review → export → retrain |

## Dependencies

- Install PyTorch separately (CUDA vs CPU builds differ — see [pytorch.org](https://pytorch.org)).
- Then `pip install -r requirements.txt`
