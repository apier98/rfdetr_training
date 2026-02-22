# RF-DETR Training Workspace

This repo is a CLI-first workspace to prepare datasets and train RF-DETR models using the `rfdetr` Python package.

## Quickstart (no install)

Run the CLI from the repo:

- `python -m rfdetr_training --help`
- `python scripts/rfdetrw.py --help`

## Typical workflow

1) Create a dataset folder:

- `python -m rfdetr_training dataset create --name "my-dataset" -c monitor`

2) Put images in `datasets/<UUID>/raw/` and YOLO labels in `datasets/<UUID>/yolo/` (optional).

3) Convert YOLO -> COCO:

- `python -m rfdetr_training dataset yolo-to-coco -d datasets/<UUID> --task seg --train-ratio 0.8 --copy-images --validate`

4) Validate COCO:

- `python -m rfdetr_training dataset validate -d datasets/<UUID> --task seg`

5) Train:

- `python -m rfdetr_training train -d datasets/<UUID> --task seg --epochs 20 --batch-size 4 --grad-accum 4 --lr 1e-4`

## Tools (optional)

The `scripts/` folder also contains standalone utilities that are useful during dataset prep/debugging, but they are not required for training:

- `scripts/infer_image.py`: run inference on a single image and visualize/inspect outputs
- `scripts/infer_webcam.py`: run webcam/live inference
- `scripts/batch_infer.py`: run inference over a folder of images
- `scripts/visualize_annotations.py`: visualize COCO annotations
- `scripts/commit_from_staging.py`: merge labeled COCO from `staging/` into a dataset split
- `scripts/remove_coco_class.py`: remove a category from COCO JSONs
- `scripts/remap_coco_ids.py`: remap COCO category ids (use carefully)
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
