# Tools (optional scripts)

This repo is CLI-first for the core workflow (dataset → validate → train), but we also keep a set of standalone scripts under `scripts/` as optional tools.

These scripts are useful for:
- debugging model checkpoints and inference outputs
- visualizing datasets and COCO annotations
- small dataset maintenance operations (category removal, id remaps)

They are intentionally not part of the “supported” training interface to keep training stable.

## Recommended usage

- Prefer `python -m rfdetr_training ...` for dataset creation/conversion/training.
- Use tools only when you need them, and treat “destructive” tools as one-off maintenance steps.
- For tools that modify COCO JSONs, keep backups (some scripts already write `.bak` files).

## Scripts

- `scripts/infer_image.py`: Single-image inference + overlay + optional JSON probe dump.
- `scripts/infer_webcam.py`: Webcam inference (display).
- `scripts/batch_infer.py`: Batch inference over a directory; writes JSON detections.
- `scripts/visualize_annotations.py`: Visualize COCO annotations by sampling images.
- `scripts/commit_from_staging.py`: Merge staged labeled COCO JSON(s) into `coco/<split>`. Note: `staging/` is not created automatically by `dataset create`; create it manually if you use this helper.
- `scripts/remove_coco_class.py`: Remove one or more `category_id` values across splits.
- `scripts/remap_coco_ids.py`: Deprecated wrapper. Prefer `python -m rfdetr_training dataset normalize-coco-ids ...`.
- `scripts/compute_detection_stats.py`: Quick per-class stats by IoU matching.
- `scripts/expand_checkpoint_head.py`: Experimental head expansion helper (finetune after).
