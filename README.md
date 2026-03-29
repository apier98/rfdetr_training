# ARIA_MoldVision

Integrated system for defect detection models, from dataset preparation to training and deployment.

## What you can do

Core (supported) CLI:
- Create dataset folders (UUID layout)
- Extract frames from multiple videos (uniform sampling)
- Convert YOLO → COCO (detect or seg)
- Validate COCO splits before training
- Subsample COCO splits (stratified by class + proportional background)
- Normalize COCO category ids to contiguous `0..N-1` (fixes common training issues)
- Train defect detection models (detect or seg)
- Export trained models to ONNX (and optionally build TensorRT engines)
- Run `doctor` to check your environment and print common fix hints

## Quickstart (no install)

Run the CLI from the repo:

- `python -m moldvision --help`
- `moldvision --help` (if installed)

## Typical workflow

1) Create a dataset folder:

- `python -m moldvision dataset create --name "my-dataset" -c monitor -c keyboard` (use multiple `-c` for multiple classes)
- `python -m moldvision dataset create --name "my-dataset" --classes-file classes.txt` (one class name per line)

1b) (Optional) Extract frames from videos for labeling:

- `python -m moldvision dataset extract-frames -v video1.mp4 -v video2.mp4 -n 500 -d datasets/<UUID>`
- This samples 500 frames uniformly across all provided videos and saves them into `datasets/<UUID>/raw/`.
- You can also use a custom output directory: `python -m moldvision dataset extract-frames -v video1.mp4 -n 100 -o my_frames/`

2) Put images in `datasets/<UUID>/raw/` and YOLO labels in `datasets/<UUID>/yolo/` (optional).


If you have **mixed labels** (some images labeled in YOLO, some already labeled in COCO):
- Keep everything in one dataset UUID folder.
- Put labels into `datasets/<UUID>/labels_inbox/`:
  - YOLO `*.txt` → `labels_inbox/yolo/`
  - COCO `_annotations.coco.json` → `labels_inbox/coco/` (images should already exist in `raw/` whenever possible)
- Run one command to ingest everything into `coco/train` + `coco/valid`:
  - `python -m moldvision dataset ingest -d datasets/<UUID> --train-ratio 0.8 --seed 0`
- The ingest step uses `METADATA.json` as the class resolver, splits by ratio, **includes unlabeled images from `raw/` as background** (empty annotations), and quarantines conflicts (multiple labels for the same image) into `labels_inbox/quarantine/`.

Important:
- If your YOLO labels are polygons (segmentation), ingest with `--yolo-task seg`.
- If you accidentally ingested with the wrong task and want a clean restart, run:
  - `python -m moldvision dataset reset-coco -d datasets/<UUID>`
  - then re-run `dataset ingest`.

Example mixed workflow:
- Merge an external COCO export into `train`:
  - `python -m moldvision dataset import-coco -d datasets/<UUID> --split train --coco-json path\\to\\_annotations.coco.json --images-dir path\\to\\images --mode copy --align-metadata`
- Convert YOLO to a temp COCO folder:
  - `python -m moldvision dataset yolo-to-coco -d datasets/<UUID> --task detect --out-dir datasets/<UUID>/exports/yolo_to_coco_tmp --copy-images`
- Merge the generated temp COCO into `train` (or `valid`):
  - `python -m moldvision dataset import-coco -d datasets/<UUID> --split train --coco-json datasets/<UUID>/exports/yolo_to_coco_tmp/train/_annotations.coco.json --images-dir datasets/<UUID>/exports/yolo_to_coco_tmp/train --mode copy --align-metadata`

3) Convert YOLO -> COCO:

- `python -m moldvision dataset yolo-to-coco -d datasets/<UUID> --task seg --train-ratio 0.8 --copy-images --validate`

4) Validate COCO:

- `python -m moldvision dataset validate -d datasets/<UUID> --task seg`

If you see warnings about 1-indexed category ids or holes, normalize ids (creates `.bak` backups):

- `python -m moldvision dataset normalize-coco-ids -d datasets/<UUID>`

If you see duplicated class names / wrong `num_classes`, align COCO categories to `METADATA.json`:

- `python -m moldvision dataset align-metadata -d datasets/<UUID>`

If you have a very large dataset and want to run quick experiments, you can subsample a split (this utility guarantees that at least one instance of every class is kept and preserves the original background image proportion):

- `python -m moldvision dataset subsample -d datasets/<UUID> --split train --fraction 0.2`

5) Train:

- `python -m moldvision train -d datasets/<UUID> --task seg --epochs 20 --batch-size 4 --grad-accum 4 --lr 1e-4`

Common training options:
- Windows stability: add `--num-workers 0`
- If you see the “Inference tensors cannot be saved for backward” error: add `--patch-inference-mode`
- Training now follows the upstream RF-DETR recommendation by default: we fine-tune from the selected model size's default pretrained weights for both detect and seg.
- To train from scratch instead, pass `--no-pretrained`.
- To override the default pretrained source, pass `--pretrain-weights path\to\weights.pth`.
- Start a new run from weights: `--finetune-from path\to\checkpoint.pth`
- Continue an existing run: `--resume path\to\checkpoint.pth`
- Evaluation only (no training): add `--eval-only`

## Troubleshooting

- `python -m moldvision doctor`

## Export (deployment)

Export to ONNX:
- `python -m moldvision export -d datasets/<UUID> -w datasets/<UUID>/models/checkpoint_best_total.pth --format onnx`
- `export` auto-detects `--task` and `--size` from `datasets/<UUID>/models/model_config.json` when available, so trained `seg` checkpoints export without restating the architecture.
- Tip: export is `--strict` by default (fails fast on mismatched checkpoints). Use `--non-strict` only for debugging.

ONNX Precision & Quantization:
- **FP16 (Half Precision)**: `python -m moldvision export -d datasets/<UUID> -w ... --format onnx_fp16`
    - Offers ~2x speedup on modern GPUs with minimal accuracy loss. No calibration needed.
- **INT8 Quantization**: `python -m moldvision export -d datasets/<UUID> -w ... --format onnx_quantized`
    - Uses **static calibration** by default using images from your dataset:
        - `--calibration-split valid`: Choose which split to use for calibration (default: `valid`).
        - `--calibration-count 100`: Number of images to use for calibration (default: `100`).
    - If no calibration data is found, it falls back to **dynamic quantization**.

Build a TensorRT engine (requires TensorRT `trtexec` on PATH):
- `python -m moldvision export -d datasets/<UUID> -w datasets/<UUID>/models/checkpoint_best_total.pth --format tensorrt --fp16`

## Deployment Bundle (portable)

After training, you can create a self-contained folder you can copy into another project. It includes:
`model.onnx` as the primary shipped model, `checkpoint.pth` as a fallback/debug artifact, `classes.json`,
`model_config.json`, `preprocess.json` (letterbox / keep aspect ratio), `postprocess.json`,
`bundle_manifest.json`, `requirements.txt`, `requirements-pytorch-fallback.txt`, and a standalone `infer.py` runner.

- Recommended: bundle the portable checkpoint source and let the bundler produce the primary ONNX artifact:
  - `python -m moldvision bundle -d datasets/<UUID> -w datasets/<UUID>/models/checkpoint_portable.pth --zip`
- **Include multiple precisions in the bundle**:
  - `python -m moldvision bundle -d datasets/<UUID> -w ... --quantize` (Adds INT8 `model_quantized.onnx`)
  - `python -m moldvision bundle -d datasets/<UUID> -w ... --export onnx_fp16` (Adds FP16 `model_fp16.onnx`)
- To also include a TensorRT engine in the bundle, add `--export tensorrt` (requires `trtexec` on `PATH`).

### Inference Hierarchy
The `infer.py` runner (and `InferenceEngine`) automatically selects the best available model in this order:
1.  **TensorRT** (`model.engine`) - Fastest (GPU only)
2.  **INT8 ONNX** (`model_quantized.onnx`) - Smallest & fast on CPU/NPU
3.  **FP16 ONNX** (`model_fp16.onnx`) - Fast on modern GPUs
4.  **FP32 ONNX** (`model.onnx`) - Baseline compatibility
5.  **PyTorch Checkpoint** (`checkpoint.pth`) - Slowest fallback

You can force a backend with `--backend {tensorrt,onnx,pytorch}`.
- You can force a backend with `--backend tensorrt`, `--backend onnx`, or `--backend pytorch`.
- If weights-only extraction fails, bundle creation now stops by default instead of silently copying the raw checkpoint. Use `--allow-raw-checkpoint-fallback` only for trusted/debug scenarios.
- Inside the bundle folder, install the primary runtime with `pip install -r requirements.txt`.
- Install `requirements-pytorch-fallback.txt` only if you want checkpoint-based fallback inference.
- Run inside the bundle folder: `python infer.py --image path\\to\\image.jpg --out-json out.json --out-image out.png`
- Segmentation: the bundle runner also supports `--mask-thresh` and `--mask-alpha` for mask overlays.

## Tools (optional)

The `scripts/` folder also contains standalone utilities that are useful during dataset prep/debugging, but they are not required for training:

- `scripts/infer_image.py`: run inference on a single image and visualize/inspect outputs
- `scripts/infer_webcam.py`: run webcam/live inference
- `scripts/infer_video.py`: run inference on a video file, draw overlays, and optionally save the result
- `scripts/batch_infer.py`: run inference over a folder of images
- `scripts/visualize_annotations.py`: visualize COCO annotations
- `scripts/commit_from_staging.py`: merge labeled COCO from `staging/` into a dataset split
- `scripts/remove_coco_class.py`: remove a category from COCO JSONs
- `scripts/remap_coco_ids.py`: deprecated wrapper (use `dataset normalize-coco-ids` instead)
- `scripts/compute_detection_stats.py`: quick TP/FP/FN stats from batch inference outputs
- `scripts/expand_checkpoint_head.py`: experimental checkpoint head expansion (use carefully)

See `docs/TOOLS.md` for notes and recommendations.

Video overlay examples:
- `python scripts/infer_video.py --video path\\to\\input.mp4 --weights datasets/<UUID>/models/checkpoint_best_total.pth --task detect --output-video runs\\video_detect.mp4`
- `python scripts/infer_video.py --video path\\to\\input.mp4 --weights datasets/<UUID>/models/checkpoint_best_total.pth --task seg --classes-file datasets/<UUID>/METADATA.json --out-dir runs\\videos`

## Install (optional)

If you want the `rfdetrw` command available globally in your venv:

- `pip install -e .`
- then run `rfdetrw --help`

Dependencies:
- Install PyTorch separately (CUDA vs CPU builds differ).
- Then `pip install -r requirements.txt`
