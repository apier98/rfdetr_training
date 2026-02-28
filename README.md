# RF-DETR Training Workspace

This repo is a CLI-first workspace to prepare datasets and train RF-DETR models using the `rfdetr` Python package.

## What you can do

Core (supported) CLI:
- Create dataset folders (UUID layout)
- Convert YOLO → COCO (detect or seg)
- Validate COCO splits before training
- Normalize COCO category ids to contiguous `0..N-1` (fixes common training issues)
- Train RF-DETR models (detect or seg)
- Export trained models to ONNX (and optionally build TensorRT engines)
- Run `doctor` to check your environment and print common fix hints

## Quickstart (no install)

Run the CLI from the repo:

- `python -m rfdetr_training --help`
- `python scripts/rfdetrw.py --help`

## Typical workflow

1) Create a dataset folder:

- `python -m rfdetr_training dataset create --name "my-dataset" -c monitor`

2) Put images in `datasets/<UUID>/raw/` and YOLO labels in `datasets/<UUID>/yolo/` (optional).

If you have **mixed labels** (some images labeled in YOLO, some already labeled in COCO):
- Keep everything in one dataset UUID folder.
- Put labels into `datasets/<UUID>/labels_inbox/`:
  - YOLO `*.txt` → `labels_inbox/yolo/`
  - COCO `_annotations.coco.json` → `labels_inbox/coco/` (images should already exist in `raw/` whenever possible)
- Run one command to ingest everything into `coco/train` + `coco/valid`:
  - `python -m rfdetr_training dataset ingest -d datasets/<UUID> --train-ratio 0.8 --seed 0`
- The ingest step uses `METADATA.json` as the class resolver, splits by ratio, **includes unlabeled images from `raw/` as background** (empty annotations), and quarantines conflicts (multiple labels for the same image) into `labels_inbox/quarantine/`.

Important:
- If your YOLO labels are polygons (segmentation), ingest with `--yolo-task seg`.
- If you accidentally ingested with the wrong task and want a clean restart, run:
  - `python -m rfdetr_training dataset reset-coco -d datasets/<UUID>`
  - then re-run `dataset ingest`.

Example mixed workflow:
- Merge an external COCO export into `train`:
  - `python -m rfdetr_training dataset import-coco -d datasets/<UUID> --split train --coco-json path\\to\\_annotations.coco.json --images-dir path\\to\\images --mode copy --align-metadata`
- Convert YOLO to a temp COCO folder:
  - `python -m rfdetr_training dataset yolo-to-coco -d datasets/<UUID> --task detect --out-dir datasets/<UUID>/exports/yolo_to_coco_tmp --copy-images`
- Merge the generated temp COCO into `train` (or `valid`):
  - `python -m rfdetr_training dataset import-coco -d datasets/<UUID> --split train --coco-json datasets/<UUID>/exports/yolo_to_coco_tmp/train/_annotations.coco.json --images-dir datasets/<UUID>/exports/yolo_to_coco_tmp/train --mode copy --align-metadata`

3) Convert YOLO -> COCO:

- `python -m rfdetr_training dataset yolo-to-coco -d datasets/<UUID> --task seg --train-ratio 0.8 --copy-images --validate`

4) Validate COCO:

- `python -m rfdetr_training dataset validate -d datasets/<UUID> --task seg`

If you see warnings about 1-indexed category ids or holes, normalize ids (creates `.bak` backups):

- `python -m rfdetr_training dataset normalize-coco-ids -d datasets/<UUID>`

If you see duplicated class names / wrong `num_classes`, align COCO categories to `METADATA.json`:

- `python -m rfdetr_training dataset align-metadata -d datasets/<UUID>`

5) Train:

- `python -m rfdetr_training train -d datasets/<UUID> --task seg --epochs 20 --batch-size 4 --grad-accum 4 --lr 1e-4`

Common training options:
- Windows stability: add `--num-workers 0`
- If you see the “Inference tensors cannot be saved for backward” error: add `--patch-inference-mode`
- Fine-tune from a pretrained RF-DETR checkpoint: add `--pretrained`
  - If `--pretrained` is set and `--pretrain-weights` is omitted, we let `rfdetr` apply its default pretrained weights behavior.
  - If `--pretrained` is not set, we default to disabling pretrained downloads/loads for reproducibility.
- Start a new run from weights: `--finetune-from path\to\checkpoint.pth`
- Continue an existing run: `--resume path\to\checkpoint.pth`
- Evaluation only (no training): add `--eval-only`

## Troubleshooting

- `python -m rfdetr_training doctor`

## Export (deployment)

Export to ONNX:
- `python -m rfdetr_training export -d datasets/<UUID> -w datasets/<UUID>/models/checkpoint_best_total.pth --format onnx`
- Tip: export is `--strict` by default (fails fast on mismatched checkpoints). Use `--non-strict` only for debugging.

Build a TensorRT engine (requires TensorRT `trtexec` on PATH):
- `python -m rfdetr_training export -d datasets/<UUID> -w datasets/<UUID>/models/checkpoint_best_total.pth --format tensorrt --fp16`

## Deployment Bundle (portable)

After training, you can create a self-contained folder you can copy into another project. It includes:
`checkpoint.pth`, `classes.json`, `model_config.json`, `preprocess.json` (letterbox / keep aspect ratio),
`postprocess.json`, and a standalone `infer.py` runner.

- Recommended: bundle the portable checkpoint (PyTorch 2.6+ friendly weights-only):
  - `python -m rfdetr_training bundle -d datasets/<UUID> -w datasets/<UUID>/models/checkpoint_portable.pth --zip`
- If you want to bundle a training checkpoint directly, the bundler will still write a weights-only `checkpoint.pth` by default:
  - `python -m rfdetr_training bundle -d datasets/<UUID> -w datasets/<UUID>/models/checkpoint_best_total.pth --zip`
  - Optional: include the original checkpoint as `checkpoint_raw.pth` (trusted/debug only) with `--include-raw-checkpoint`.
- Run inside the bundle folder: `python infer.py --image path\\to\\image.jpg --out-json out.json --out-image out.png`
- Segmentation: the bundle runner also supports `--mask-thresh` and `--mask-alpha` for mask overlays.

## Tools (optional)

The `scripts/` folder also contains standalone utilities that are useful during dataset prep/debugging, but they are not required for training:

- `scripts/infer_image.py`: run inference on a single image and visualize/inspect outputs
- `scripts/infer_webcam.py`: run webcam/live inference
- `scripts/batch_infer.py`: run inference over a folder of images
- `scripts/visualize_annotations.py`: visualize COCO annotations
- `scripts/commit_from_staging.py`: merge labeled COCO from `staging/` into a dataset split
- `scripts/remove_coco_class.py`: remove a category from COCO JSONs
- `scripts/remap_coco_ids.py`: deprecated wrapper (use `dataset normalize-coco-ids` instead)
- `scripts/compute_detection_stats.py`: quick TP/FP/FN stats from batch inference outputs
- `scripts/expand_checkpoint_head.py`: experimental checkpoint head expansion (use carefully)

See `docs/TOOLS.md` for notes and recommendations.

## Install (optional)

If you want the `rfdetrw` command available globally in your venv:

- `pip install -e .`
- then run `rfdetrw --help`

Dependencies:
- Install PyTorch separately (CUDA vs CPU builds differ).
- Then `pip install -r requirements.txt`
