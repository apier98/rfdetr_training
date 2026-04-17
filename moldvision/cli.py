from __future__ import annotations

import argparse
import sys
from typing import List

from . import appconfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="moldvision", description="ARIA_MoldVision: Defect Detection workspace (CLI)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # doctor
    sub.add_parser("doctor", help="Check environment and print common fix hints")

    # export
    ex = sub.add_parser("export", help="Export a trained model for deployment (ONNX / TensorRT)")
    ex.add_argument("--dataset-dir", "-d", required=True)
    ex.add_argument("--weights", "-w", required=True, help="Checkpoint path (.pth)")
    ex.add_argument(
        "--task",
        choices=["detect", "seg"],
        default=None,
        help="Task for export. Default: auto (from datasets/<UUID>/models/model_config.json).",
    )
    ex.add_argument(
        "--size",
        choices=["nano", "small", "base", "medium", "large", "xlarge", "2xlarge"],
        default=None,
        help="Model size preset. Default: auto (from datasets/<UUID>/models/model_config.json). Ignored if using checkpoint model.",
    )
    ex.add_argument("--format", choices=["onnx", "tensorrt", "onnx_quantized", "onnx_fp16"], default=None, help="Export format (default: from config/env, fallback onnx)")
    ex.add_argument("--output", "-o", default=None, help="Output path (.onnx or .engine). Default: datasets/<UUID>/exports/")
    ex.add_argument("--device", default=None, help="Device for export (e.g. cuda:0, cpu)")
    ex.add_argument("--height", type=int, default=640)
    ex.add_argument("--width", type=int, default=640)
    ex.add_argument("--opset", type=int, default=18, help="ONNX opset")
    ex.add_argument("--dynamic", action="store_true", help="Dynamic input H/W axes (ONNX only)")
    ex.add_argument("--quantize", action="store_true", help="Enable ONNX quantization (for --format onnx_quantized)")
    ex.add_argument("--calibration-split", default="valid", help="Dataset split for quantization calibration")
    ex.add_argument("--calibration-count", type=int, default=100, help="Number of images for calibration")
    ex.add_argument("--use-checkpoint-model", action="store_true", help="If checkpoint contains a pickled model, use it (trusted only)")
    ex.add_argument("--checkpoint-key", default=None, help="Explicit key inside checkpoint that contains state_dict")
    ex.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Fail if checkpoint does not match model exactly (recommended for deployment). Default: enabled.",
    )
    ex.add_argument(
        "--non-strict",
        dest="strict",
        action="store_false",
        help="Allow partial checkpoint loads (debugging only; can cause subtle deployment bugs).",
    )
    ex.add_argument("--fp16", action="store_true", help="TensorRT: build FP16 engine")
    ex.add_argument("--workspace-mb", type=int, default=2048, help="TensorRT: workspace size in MB")

    # bundle
    bd = sub.add_parser("bundle", help="Create a portable deployment bundle (checkpoint + configs + runner)")
    bd.add_argument("--dataset-dir", "-d", required=True)
    bd.add_argument("--weights", "-w", required=True, help="Checkpoint path (.pth)")
    bd.add_argument(
        "--task",
        choices=["detect", "seg"],
        default=None,
        help="Task for the bundle. Default: auto (from datasets/<UUID>/models/model_config.json)",
    )
    bd.add_argument(
        "--size",
        choices=["nano", "small", "base", "medium", "large", "xlarge", "2xlarge"],
        default=None,
        help="Model size preset. Default: auto (from datasets/<UUID>/models/model_config.json)",
    )
    bd.add_argument("--output-dir", "-o", default=None, help="Output bundle directory (default: datasets/<UUID>/deploy/<weights>_<timestamp>)")
    bd.add_argument("--height", type=int, default=None, help="Model input height (default: trained resolution or 640)")
    bd.add_argument("--width", type=int, default=None, help="Model input width (default: trained resolution or 640)")
    bd.add_argument(
        "--export",
        action="append",
        default=None,
        choices=["onnx", "tensorrt", "onnx_quantized", "onnx_fp16"],
        help="Extra exported formats to include. Default behavior already ships ONNX as the primary bundle artifact.",
    )
    bd.add_argument("--opset", type=int, default=18, help="ONNX opset")
    bd.add_argument("--dynamic", action="store_true", help="ONNX: export dynamic H/W axes")
    bd.add_argument("--quantize", action="store_true", help="Include quantized ONNX model in bundle")
    bd.add_argument("--calibration-split", default="valid", help="Dataset split for quantization calibration")
    bd.add_argument("--calibration-count", type=int, default=100, help="Number of images for calibration")
    bd.add_argument("--device", default=None, help="Device for export (e.g. cuda:0, cpu)")
    bd.add_argument("--use-checkpoint-model", action="store_true", help="If checkpoint contains a pickled model, use it (trusted only)")
    bd.add_argument("--checkpoint-key", default=None, help="Explicit key inside checkpoint that contains state_dict")
    pc = bd.add_mutually_exclusive_group()
    pc.add_argument(
        "--portable-checkpoint",
        dest="portable_checkpoint",
        action="store_true",
        default=True,
        help="Write a weights-only (tensor-only state_dict) checkpoint into the bundle (recommended). Default: enabled.",
    )
    pc.add_argument(
        "--no-portable-checkpoint",
        dest="portable_checkpoint",
        action="store_false",
        help="Copy the checkpoint verbatim into the bundle (may require unsafe torch.load weights_only=False on some PyTorch versions).",
    )
    bd.add_argument(
        "--allow-raw-checkpoint-fallback",
        action="store_true",
        help="If weights-only checkpoint extraction fails, copy the raw checkpoint into the bundle instead (trusted/debug only).",
    )
    bd.add_argument(
        "--include-raw-checkpoint",
        action="store_true",
        help="Also store the original checkpoint as checkpoint_raw.pth (debugging/trusted only).",
    )
    bd.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Fail if checkpoint does not match model exactly (recommended for deployment). Default: enabled.",
    )
    bd.add_argument("--non-strict", dest="strict", action="store_false", help="Allow partial checkpoint loads (debugging only).")
    bd.add_argument("--fp16", action="store_true", help="TensorRT: build FP16 engine")
    bd.add_argument("--workspace-mb", type=int, default=2048, help="TensorRT: workspace size in MB")
    bd.add_argument("--zip", action="store_true", help="Also write <bundle>.zip next to the bundle dir")
    bd.add_argument("--mpk", action="store_true", help="Also write <bundle>.mpk next to the bundle dir (ARIA MoldPilot install format)")
    bd.add_argument("--overwrite", action="store_true", help="Overwrite output dir if it exists")
    # MoldPilot identity fields
    bd.add_argument("--bundle-id", default=None, help="Unique bundle identifier (default: auto-generated from model-name + version, e.g. mymodel-v1.0.0)")
    bd.add_argument("--model-name", default=None, help="Human-readable model name (default: dataset name)")
    bd.add_argument("--model-version", default="1.0.0", help="Semantic version string (default: 1.0.0)")
    bd.add_argument("--channel", default="stable", choices=["stable", "beta"], help="Release channel (default: stable)")
    bd.add_argument("--supersedes", default=None, help="bundle_id of the previous version this release replaces (for MoldPilot rollback support)")
    bd.add_argument("--min-app-version", default="0.0.0", help="Minimum ARIA MoldPilot version required to use this bundle (default: 0.0.0)")
    bd.add_argument(
        "--standalone",
        action="store_true",
        help="Include Python inference runner (infer.py, requirements.txt, vendored moldvision/) for use outside of ARIA MoldPilot.",
    )
    bd.add_argument(
        "--publish",
        action="store_true",
        help="Immediately publish the bundle to the S3 model catalog after creation.",
    )
    bd.add_argument("--publish-role", default=None, help="Model role for publish (e.g., defect_detector). Required with --publish.")
    bd.add_argument("--publish-dry-run", action="store_true", help="Dry-run publish (print catalog entry without uploading)")

    # infer (bundle)
    inf = sub.add_parser("infer", help="Run inference using a deployment bundle directory (debug/smoke check)")
    inf.add_argument("--bundle-dir", "-b", required=True)
    _inf_src = inf.add_mutually_exclusive_group(required=True)
    _inf_src.add_argument("--image", "-i", default=None, help="Single image path for inference")
    _inf_src.add_argument("--input-dir", default=None, help="Directory of images for batch inference")
    inf.add_argument("--out-dir", default=None, help="Batch mode: output directory for JSON results and overlays (default: <input-dir>/results/)")
    inf.add_argument("--overlays", action="store_true", help="Batch mode: also write overlay images alongside JSON results")
    inf.add_argument("--weights", "-w", default=None, help="Override fallback checkpoint path (default: <bundle>/checkpoint.pth)")
    inf.add_argument("--backend", choices=["auto", "tensorrt", "onnx", "pytorch"], default=None, help="Inference backend (default: from config/env, fallback auto)")
    inf.add_argument("--device", default=None)
    inf.add_argument("--threshold", type=float, default=None)
    inf.add_argument("--mask-thresh", type=float, default=None)
    inf.add_argument("--mask-alpha", type=float, default=None, help="Segmentation: mask overlay alpha (0..1)")
    inf.add_argument("--use-checkpoint-model", action="store_true", help="If checkpoint contains a pickled model, use it (trusted only)")
    inf.add_argument("--checkpoint-key", default=None)
    inf.add_argument("--strict", action="store_true", default=False, help="Strict state_dict load (recommended for deployment bundles)")
    inf.add_argument("--out-json", default=None)
    inf.add_argument("--out-image", default=None, help="Optional: write an overlay image (boxes + masks if seg)")

    # dataset
    ds = sub.add_parser("dataset", help="Dataset utilities")
    ds_sub = ds.add_subparsers(dest="dataset_cmd", required=True)

    ds_create = ds_sub.add_parser("create", help="Create a dataset UUID folder")
    ds_create.add_argument("--uuid", "-u", default=None)
    ds_create.add_argument("--root", "-r", default=None, help="Root directory for new datasets (default: resolved from config/env)")
    ds_create.add_argument("--name", "-n", default=None)
    ds_create.add_argument("--force", "-f", action="store_true")
    ds_create.add_argument("--no-readme", action="store_true")
    ds_create.add_argument("--classes", "-c", action="append", default=None)
    ds_create.add_argument("--classes-file", default=None)

    ds_y2c = ds_sub.add_parser("yolo-to-coco", help="Convert YOLO labels to COCO train/valid splits")
    ds_y2c.add_argument("--dataset-dir", "-d", required=True)
    ds_y2c.add_argument("--task", choices=["detect", "seg"], default="detect")
    ds_y2c.add_argument("--train-ratio", type=float, default=0.8)
    ds_y2c.add_argument("--seed", type=int, default=0)
    ds_y2c.add_argument("--copy-images", action="store_true")
    ds_y2c.add_argument("--images-ext", default="jpg,png")
    ds_y2c.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for generated COCO (default: <dataset>/coco). Useful for mixed COCO+YOLO workflows.",
    )
    ds_y2c.add_argument("--validate", action="store_true")
    ds_y2c.add_argument("--validate-only", action="store_true")

    ds_val = ds_sub.add_parser("validate", help="Validate COCO split files")
    ds_val.add_argument("--dataset-dir", "-d", required=True)
    ds_val.add_argument("--task", choices=["detect", "seg"], default="detect")
    ds_val.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    ds_val.add_argument("--check-images", action="store_true", help="Warn if image files are missing from split dir")

    ds_prune = ds_sub.add_parser(
        "prune-empty-masks",
        help="Segmentation: drop invalid/empty masks and remove images that have 0 masks (fixes 'masks cannot be empty')",
    )
    ds_prune.add_argument("--dataset-dir", "-d", required=True)
    ds_prune.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    ds_prune.add_argument("--dry-run", action="store_true")

    ds_prune_small = ds_sub.add_parser(
        "prune-small-masks",
        help="Segmentation: remove tiny instances that may vanish after resizing (prevents intermittent 'masks cannot be empty')",
    )
    ds_prune_small.add_argument("--dataset-dir", "-d", required=True)
    ds_prune_small.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    ds_prune_small.add_argument("--resolution", type=int, required=True, help="Square resize resolution used for training (e.g. 312/448)")
    ds_prune_small.add_argument("--min-scaled-area", type=float, default=1.0, help="Min instance area after scaling to resolution (pixels^2)")
    ds_prune_small.add_argument("--dry-run", action="store_true")

    ds_norm = ds_sub.add_parser("normalize-coco-ids", help="Normalize COCO category ids to contiguous 0..N-1 (safe backup)")
    ds_norm.add_argument("--dataset-dir", "-d", required=True)
    ds_norm.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    ds_norm.add_argument("--dry-run", action="store_true")

    ds_align = ds_sub.add_parser("align-metadata", help="Align COCO categories to METADATA.json class_names (fix wrong num_classes)")
    ds_align.add_argument("--dataset-dir", "-d", required=True)
    ds_align.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    ds_align.add_argument("--dry-run", action="store_true")

    ds_reset = ds_sub.add_parser("reset-coco", help="Reset coco/ splits to empty (backs up existing coco/)")
    ds_reset.add_argument("--dataset-dir", "-d", required=True)
    ds_reset.add_argument("--no-backup", action="store_true", help="Do not move existing coco/ to a backup folder")

    ds_subsample = ds_sub.add_parser("subsample", help="Subsample a COCO split ensuring class representation and proportional background images")
    ds_subsample.add_argument("--dataset-dir", "-d", required=True)
    ds_subsample.add_argument("--split", choices=["train", "valid", "test", "all"], required=True,
                              help="Split to subsample, or 'all' to run on train/valid/test in sequence")
    ds_subsample.add_argument("--fraction", type=float, default=None, help="Fraction of images to keep (e.g., 0.25)")
    ds_subsample.add_argument("--max-images", type=int, default=None, help="Maximum number of images to keep")
    ds_subsample.add_argument("--min-instances", type=int, default=1, help="Minimum instances per class to keep")
    ds_subsample.add_argument("--seed", type=int, default=42, help="Random seed")
    ds_subsample.add_argument("--dry-run", action="store_true")

    ds_imp = ds_sub.add_parser("import-coco", help="Import/merge an external COCO JSON (and images) into this dataset split")
    ds_imp.add_argument("--dataset-dir", "-d", required=True)
    ds_imp.add_argument("--split", choices=["train", "valid", "test"], default="train")
    ds_imp.add_argument("--coco-json", required=True, help="Path to source _annotations.coco.json")
    ds_imp.add_argument("--images-dir", default=None, help="Folder containing images referenced by COCO file_name")
    ds_imp.add_argument("--mode", choices=["copy", "move"], default="copy")
    ds_imp.add_argument("--rename", action="store_true", help="Rename imported images to sequential numbers to avoid collisions")
    ds_imp.add_argument("--pad", type=int, default=6, help="Zero-pad width when renaming (default: 6)")
    ds_imp.add_argument("--align-metadata", action="store_true", help="Map categories to METADATA.json class_names by name")
    ds_imp.add_argument("--dry-run", action="store_true")

    ds_ing = ds_sub.add_parser("ingest", help="Ingest mixed labels from labels_inbox/ into coco/ (YOLO + COCO)")
    ds_ing.add_argument("--dataset-dir", "-d", required=True)
    ds_ing.add_argument("--train-ratio", type=float, default=0.8)
    ds_ing.add_argument("--seed", type=int, default=0)
    ds_ing.add_argument(
        "--yolo-task",
        "--task",
        dest="yolo_task",
        choices=["detect", "seg"],
        default="detect",
        help="How to interpret YOLO labels during conversion",
    )
    ds_ing.add_argument("--images-ext", default="jpg,png", help="Comma-separated image extensions under raw/ (default: jpg,png)")
    ds_ing.add_argument("--mode", choices=["copy", "move"], default="copy", help="Copy/move images into coco splits (default: copy)")
    ds_ing.add_argument(
        "--include-background",
        action="store_true",
        default=True,
        help="Include unlabeled images from raw/ as background (empty annotations). Default: enabled.",
    )
    ds_ing.add_argument(
        "--no-include-background",
        dest="include_background",
        action="store_false",
        help="Do not include unlabeled images from raw/.",
    )
    ds_ing.add_argument("--no-align-metadata", dest="align_metadata", action="store_false", help="Do not align COCO categories to METADATA.json")
    ds_ing.set_defaults(align_metadata=True)
    ds_ing.add_argument("--dry-run", action="store_true")

    ds_ext = ds_sub.add_parser("extract-frames", help="Extract frames from video files for labeling")
    ds_ext.add_argument("--videos", "-v", action="append", default=None,
                        help="Path to a video file (can be repeated for multiple files)")
    ds_ext.add_argument("--videos-dir", default=None,
                        help="Folder to scan for video files (combined with --ext filter)")
    ds_ext.add_argument("--ext", default="mp4,avi,mov,mkv,webm",
                        help="Comma-separated video extensions to scan when using --videos-dir (default: mp4,avi,mov,mkv,webm)")
    ds_ext.add_argument("--num-frames", "-n", type=int, default=None,
                        help="Total number of frames to extract across all videos")
    ds_ext.add_argument("--fps", type=float, default=None,
                        help="Extract this many frames per second of video (alternative to --num-frames)")
    ds_ext.add_argument("--dataset-dir", "-d", default=None,
                        help="Dataset folder; frames are saved in <dataset-dir>/raw/")
    ds_ext.add_argument("--out-dir", "-o", default=None,
                        help="Custom output directory (overrides --dataset-dir)")

    ds_list = ds_sub.add_parser("list", help="List all datasets under the configured root directory")
    ds_list.add_argument("--root", "-r", default=None, help="Root directory to scan (default: resolved from config/env)")

    ds_info = ds_sub.add_parser("info", help="Show detailed information about a dataset")
    ds_info.add_argument("--dataset-dir", "-d", required=True, help="Path to a dataset UUID folder")

    # train
    tr = sub.add_parser("train", help="Train an RF-DETR model")
    tr.add_argument("--dataset-dir", "-d", required=True)
    tr.add_argument("--task", choices=["detect", "seg"], default="detect")
    tr.add_argument(
        "--size",
        choices=["nano", "small", "base", "medium", "large", "xlarge", "2xlarge"],
        default="nano",
        help="Model size preset (some sizes may require rfdetr[plus])",
    )
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--batch-size", type=int, default=4)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--device", default=None, help="Device string passed to RF-DETR trainer (e.g. cuda, cuda:0, cpu)")
    tr.add_argument("--num-workers", type=int, default=None, help="Dataloader workers (Windows recommended: 0)")
    tr.add_argument("--resolution", type=int, default=None, help="Training resolution; must be divisible by 32 (often 224 works well)")
    tr.add_argument("--output-dir", "-o", default=None)
    pre = tr.add_mutually_exclusive_group()
    pre.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,
        help="Use the upstream RF-DETR default pretrained weights for the selected model size. Default: enabled.",
    )
    pre.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable upstream pretrained initialization and start from scratch.",
    )
    tr.add_argument("--pretrain-weights", default=None)
    tr.add_argument("--tensorboard", action="store_true")
    tr.add_argument("--wandb", action="store_true")
    tr.add_argument("--early-stopping", action="store_true")
    tr.add_argument(
        "--eval-only",
        "--eval",
        dest="eval_only",
        action="store_true",
        help="Run evaluation only and exit (no training).",
    )
    tr.add_argument("--num-queries", type=int, default=None)
    tr.add_argument("--num-select", type=int, default=None)
    tr.add_argument("--run-test", action="store_true")
    tr.add_argument("--benchmark", action="store_true")
    tr.add_argument("--no-aug", action="store_true", help="Disable augmentations (sets aug_config={})")
    tr.add_argument("--aug-config", default=None, help="Path to JSON/YAML augmentation config (advanced)")
    ms = tr.add_mutually_exclusive_group()
    ms.add_argument("--multi-scale", dest="multi_scale", action="store_true", default=None, help="Enable multi-scale training")
    ms.add_argument("--no-multi-scale", dest="multi_scale", action="store_false", help="Disable multi-scale training")
    es = tr.add_mutually_exclusive_group()
    es.add_argument("--expanded-scales", dest="expanded_scales", action="store_true", default=None, help="Enable expanded multi-scale ranges")
    es.add_argument("--no-expanded-scales", dest="expanded_scales", action="store_false", help="Disable expanded multi-scale ranges")
    rp = tr.add_mutually_exclusive_group()
    rp.add_argument(
        "--random-resize-via-padding",
        dest="do_random_resize_via_padding",
        action="store_true",
        default=None,
        help="Enable random resize via padding (advanced; may affect masks)",
    )
    rp.add_argument(
        "--no-random-resize-via-padding",
        dest="do_random_resize_via_padding",
        action="store_false",
        help="Disable random resize via padding",
    )

    # safer replacements than the old "resume but partially load and then remove resume"
    tr.add_argument("--resume", default=None, help="Resume training (trainer checkpoint)")
    tr.add_argument("--finetune-from", default=None,
        help="Bundle ID (or raw path) to fine-tune from. Resolves checkpoint from lake model registry.")
    tr.add_argument("--use-checkpoint-model", action="store_true", help="If checkpoint contains a pickled model, use it (trusted only)")
    tr.add_argument("--checkpoint-key", default=None, help="Explicit key inside checkpoint that contains state_dict")

    tr.add_argument(
        "--patch-inference-mode",
        action="store_true",
        default=None,
        help="Workaround: patch torch.inference_mode -> enable_grad (default: auto on Windows)",
    )
    tr.add_argument(
        "--no-patch-inference-mode",
        dest="patch_inference_mode",
        action="store_false",
        help="Disable inference_mode patch (if auto-enabled)",
    )
    tr.add_argument("--no-validate", action="store_true", help="Skip dataset validation checks")
    tr.add_argument("--log-file", default=None, help="Write training output to this log file (default: auto → <output_dir>/training.log)")
    tr.add_argument("--no-log-file", action="store_true", help="Disable file logging (console only)")

    # config
    cfg_p = sub.add_parser("config", help="View or edit persistent MoldVision settings")
    cfg_sub = cfg_p.add_subparsers(dest="config_cmd", required=True)

    cfg_sub.add_parser("show", help="Print all current settings and their effective values")

    cfg_set = cfg_sub.add_parser("set", help="Persist a setting to the config file")
    cfg_set_sub = cfg_set.add_subparsers(dest="config_set_cmd", required=True)

    cfg_ds = cfg_set_sub.add_parser("dataset-root", help="Set the default dataset root directory")
    cfg_ds.add_argument("path", help="Absolute or ~ path to use as the default dataset root")

    cfg_nw = cfg_set_sub.add_parser("num-workers", help="Set the default DataLoader worker count")
    cfg_nw.add_argument("value", type=int, help="Number of workers (0 recommended on Windows)")

    cfg_bk = cfg_set_sub.add_parser("inference-backend", help="Set the preferred inference backend")
    cfg_bk.add_argument("value", choices=list(appconfig.VALID_BACKENDS), help="Backend name")

    cfg_ef = cfg_set_sub.add_parser("export-format", help="Set the default export format")
    cfg_ef.add_argument("value", choices=list(appconfig.VALID_EXPORT_FORMATS), help="Format name")

    cfg_pr = cfg_set_sub.add_parser("predictive-runs-root", help="Set the default local root for predictive training runs")
    cfg_pr.add_argument("path", help="Absolute or ~ path to use for predictive train outputs when --output-dir is omitted")

    # -------------------------------------------------------------------------
    # lake  (ARIA Data Lake management)
    # -------------------------------------------------------------------------
    lk = sub.add_parser("lake", help="ARIA Data Lake: session import, labeling, pull, model registry")
    lk_sub = lk.add_subparsers(dest="lake_cmd", required=True)

    # lake init
    lk_init = lk_sub.add_parser("init", help="Initialise a new data lake at the given path")
    lk_init.add_argument("--root", default=None, help="Lake root directory (default: ARIA_DATA_LAKE env or platform default)")

    # lake import  (external pre-labeled or unlabeled data, not from MoldPilot)
    lk_ext = lk_sub.add_parser(
        "import",
        help="Import externally-sourced images (and optional pre-existing annotations) as a synthetic session",
    )
    lk_ext.add_argument("--images-dir", required=True, help="Directory of JPEG/PNG images to import")
    lk_ext.add_argument("--task", choices=["detect", "seg"], required=True,
                        help="'detect' → inspection frames  |  'seg' → monitor frames")
    lk_ext.add_argument("--coco-json", default=None,
                        help="Optional COCO annotation file. Annotated images are marked 'labeled'; "
                             "remaining images are marked 'unlabeled' (partial annotation is fine).")
    lk_ext.add_argument("--session-id", default=None,
                        help="Custom session ID (default: auto-generated 'external_<ts>_<uuid>')")
    lk_ext.add_argument("--name", default=None, help="Human-readable name for this data source")
    lk_ext.add_argument("--machine-id", default=None)
    lk_ext.add_argument("--mold-id", default=None)
    lk_ext.add_argument("--part-id", default=None)
    lk_ext.add_argument("--notes", default=None, help="Freeform notes (data origin, supplier, etc.)")
    lk_ext.add_argument("--overwrite", action="store_true", help="Overwrite if session ID already exists")
    lk_ext.add_argument("--lake-root", default=None)

    # lake session
    lk_sess = lk_sub.add_parser("session", help="Session-level commands (import, list)")
    lk_sess_sub = lk_sess.add_subparsers(dest="lake_session_cmd", required=True)

    lk_imp = lk_sess_sub.add_parser("import", help="Import a qual session into the lake (from extracted frames or directly from videos)")
    lk_imp.add_argument("--session-meta", default=None, help="Path to session JSON (MoldPilot manifest or MoldTrace session.json)")
    lk_imp.add_argument("--session-dir", default=None,
                        help="MoldPilot session folder containing session.json and chunk videos (auto-detects streams)")
    lk_imp.add_argument("--inspection-frames", default=None, help="Directory of extracted inspection JPEGs")
    lk_imp.add_argument("--monitor-frames", default=None, help="Directory of extracted monitor JPEGs")
    lk_imp.add_argument("--inspection-videos-dir", default=None, help="Directory of inspection videos to extract from (e.g., component_view_chunk_*.mp4)")
    lk_imp.add_argument("--monitor-videos-dir", default=None, help="Directory of monitor/process videos to extract from")
    lk_imp.add_argument("--video-ext", default="mp4,avi,mov,mkv,webm,m4v,wmv,flv",
                        help="Comma-separated video extensions to scan in video dirs (default: mp4,avi,mov,mkv,webm,m4v,wmv,flv)")
    lk_imp.add_argument("--extract-fps", type=float, default=0.5,
                        help="Target extraction FPS when importing from videos (default: 0.5)")
    lk_imp.add_argument("--extract-frames", type=int, default=None,
                        help="Exact total frames to extract per stream when importing from videos (overrides --extract-fps)")
    lk_imp.add_argument("--overwrite", action="store_true", help="Overwrite if session already exists")
    lk_imp.add_argument("--lake-root", default=None, help="Override lake root path")

    lk_lst = lk_sess_sub.add_parser("list", help="List sessions with frame and annotation coverage")
    lk_lst.add_argument("--machine-id", default=None)
    lk_lst.add_argument("--mold-id", default=None)
    lk_lst.add_argument("--part-id", default=None)
    lk_lst.add_argument("--from", dest="from_date", default=None, metavar="DATE", help="Filter sessions starting on or after DATE (ISO-8601)")
    lk_lst.add_argument("--to", dest="to_date", default=None, metavar="DATE", help="Filter sessions starting on or before DATE (ISO-8601)")
    lk_lst.add_argument("--task", choices=["detect", "seg"], default=None, help="Show coverage for this task")
    lk_lst.add_argument("--label-status", choices=["labeled", "unlabeled", "any"], default=None)
    lk_lst.add_argument("--marker", default=None, help="Only sessions containing this marker text")
    lk_lst.add_argument("--min-frames", type=int, default=0, help="Minimum total frames to include a session")
    lk_lst.add_argument("--lake-root", default=None)

    lk_mbg = lk_sess_sub.add_parser(
        "mark-bg",
        help="Mark all unlabeled frames in a session as background (confirmed negatives)",
    )
    lk_mbg.add_argument("--session", required=True, help="Session ID to process")
    lk_mbg.add_argument("--task", choices=["detect", "seg"], required=True)
    lk_mbg.add_argument("--dry-run", action="store_true", help="Preview without modifying the index")
    lk_mbg.add_argument("--lake-root", default=None)

    # lake label-batch
    lk_lb = lk_sub.add_parser("label-batch", help="Labeling batch commands (create, commit, status)")
    lk_lb_sub = lk_lb.add_subparsers(dest="lake_label_batch_cmd", required=True)

    lk_lbc = lk_lb_sub.add_parser("create", help="Select frames from the lake and create a Label Studio batch")
    lk_lbc.add_argument("--task", choices=["detect", "seg"], required=True)
    lk_lbc.add_argument("--sessions", default=None, help="Comma-separated session IDs")
    lk_lbc.add_argument("--all", dest="all_sessions", action="store_true", help="Include all sessions")
    lk_lbc.add_argument("--machine-id", default=None)
    lk_lbc.add_argument("--mold-id", default=None)
    lk_lbc.add_argument("--marker", default=None)
    lk_lbc.add_argument("--only-unlabeled", dest="only_unlabeled", action="store_true", default=True)
    lk_lbc.add_argument("--include-labeled", dest="only_unlabeled", action="store_false",
                        help="Include already-labeled frames in selection pool")
    lk_lbc.add_argument("--n", type=int, default=200, help="Total frames to select across sessions (default: 200)")
    lk_lbc.add_argument("--sample-mode", choices=["random", "temporal"], default="random",
                        help="Frame selection strategy (default: random)")
    lk_lbc.add_argument("--min-frame-gap", type=int, default=1,
                        help="Minimum frame_idx gap between selected frames (avoids near-duplicates, default: 1)")
    lk_lbc.add_argument("--skip-first", type=int, default=0, help="Skip first N frames per session (startup noise)")
    lk_lbc.add_argument("--skip-last", type=int, default=0, help="Skip last N frames per session (teardown noise)")
    lk_lbc.add_argument("--seed", type=int, default=42)
    lk_lbc.add_argument("--batch-name", default=None, help="Human-readable batch name (used as ID prefix)")
    lk_lbc.add_argument("--lake-root", default=None)

    lk_lbk = lk_lb_sub.add_parser("commit", help="Commit a Label Studio COCO export back into the lake")
    lk_lbk.add_argument("--batch", required=True, dest="batch_id", help="Batch ID to commit")
    lk_lbk.add_argument("--coco-json", default=None, help="Explicit path to COCO JSON (default: auto-detect in batch/export/)")
    lk_lbk.add_argument("--ls-url", default=None, help="Label Studio base URL (e.g. http://localhost:8080)")
    lk_lbk.add_argument("--ls-api-token", default=None, help="Label Studio API token (fallback: LABEL_STUDIO_API_TOKEN)")
    lk_lbk.add_argument("--ls-project-id", type=int, default=None, help="Label Studio project ID (fallback: LABEL_STUDIO_PROJECT_ID)")
    lk_lbk.add_argument("--ls-export-type", default="COCO", help="Label Studio export type (default: COCO)")
    lk_lbk.add_argument("--dry-run", action="store_true")
    lk_lbk.add_argument("--lake-root", default=None)

    lk_lbs = lk_lb_sub.add_parser("status", help="List all label batches and their status")
    lk_lbs.add_argument("--task", choices=["detect", "seg"], default=None)
    lk_lbs.add_argument("--lake-root", default=None)

    # lake pull
    lk_pull = lk_sub.add_parser("pull", help="Build a balanced training dataset from labeled lake images")
    lk_pull.add_argument("--task", choices=["detect", "seg"], required=True)
    lk_pull.add_argument("--sessions", default=None, help="Comma-separated session IDs")
    lk_pull.add_argument("--all", dest="all_sessions", action="store_true")
    lk_pull.add_argument("--machine-id", default=None)
    lk_pull.add_argument("--mold-id", default=None)
    lk_pull.add_argument("--part-id", default=None)
    lk_pull.add_argument("--from", dest="from_date", default=None, metavar="DATE")
    lk_pull.add_argument("--to", dest="to_date", default=None, metavar="DATE")
    lk_pull.add_argument("--marker", default=None)
    lk_pull.add_argument("--include-hard-negatives", action="store_true")
    lk_pull.add_argument("--include-backgrounds", action="store_true")
    lk_pull.add_argument("--total", type=int, default=None,
                         help="Global cap on total images across all sessions (after per-session caps)")
    lk_pull.add_argument("--max-per-session", type=int, default=None)
    lk_pull.add_argument("--min-per-session", type=int, default=0)
    lk_pull.add_argument(
        "--priority-sessions", default=None,
        help="Comma-separated session IDs that get full --max-per-session quota; others are capped at max/weight",
    )
    lk_pull.add_argument(
        "--priority-weight", type=float, default=3.0,
        help="Non-priority cap divisor (default 3 → they get max_per_session / 3)",
    )
    lk_pull.add_argument("--balance-classes", action="store_true")
    lk_pull.add_argument("--min-per-class", type=int, default=None)
    lk_pull.add_argument("--train-ratio", type=float, default=0.8)
    lk_pull.add_argument("--seed", type=int, default=42)
    lk_pull.add_argument("--dataset-uuid", default=None)
    lk_pull.add_argument("--dataset-name", default=None)
    lk_pull.add_argument("--dataset-root", default=None, help="Override dataset output root (default: <lake>/datasets/)")
    lk_pull.add_argument("--dry-run", action="store_true")
    lk_pull.add_argument("--lake-root", default=None)

    # lake index
    lk_idx = lk_sub.add_parser("index", help="Manage the image_index.jsonl catalogue")
    lk_idx_grp = lk_idx.add_mutually_exclusive_group(required=True)
    lk_idx_grp.add_argument("--rebuild", action="store_true", help="Full scan → rewrite image_index.jsonl")
    lk_idx_grp.add_argument("--stats", action="store_true", help="Print per-session coverage statistics")
    lk_idx.add_argument("--task", choices=["detect", "seg"], default=None)
    lk_idx.add_argument("--lake-root", default=None)

    # lake models
    lk_mod = lk_sub.add_parser("models", help="Model registry commands (install, list, promote)")
    lk_mod_sub = lk_mod.add_subparsers(dest="lake_models_cmd", required=True)

    lk_mi = lk_mod_sub.add_parser("install", help="Install a .mpk bundle into the registry")
    lk_mi.add_argument("bundle", help="Path to .mpk file")
    lk_mi.add_argument("--task", choices=["detect", "seg"], required=True)
    lk_mi.add_argument("--lake-root", default=None)

    lk_ml = lk_mod_sub.add_parser("list", help="List installed model bundles")
    lk_ml.add_argument("--task", choices=["detect", "seg"], default=None)
    lk_ml.add_argument("--lake-root", default=None)

    lk_mp = lk_mod_sub.add_parser("promote", help="Set a bundle as the active model for a channel")
    lk_mp.add_argument("bundle_id", help="Bundle ID to promote")
    lk_mp.add_argument("--task", choices=["detect", "seg"], required=True)
    lk_mp.add_argument("--channel", choices=["stable", "dev"], required=True)
    lk_mp.add_argument("--lake-root", default=None)

    # lake pools
    lk_pools = lk_sub.add_parser("pools", help="Manage hard-negative and background pools")
    lk_pools_sub = lk_pools.add_subparsers(dest="lake_pools_cmd", required=True)

    lk_phn = lk_pools_sub.add_parser("add-hard-negative", help="Add image(s) to the hard-negatives pool")
    lk_phn.add_argument("--image", action="append", default=[], dest="images",
                        help="Relative path to image in the lake (can be repeated)")
    lk_phn.add_argument("--reason", default="", help="Why this is a hard negative (e.g. 'model_false_positive')")
    lk_phn.add_argument("--task", choices=["detect", "seg"], default=None,
                        help="Force index status update for this task. Auto-detected from path if omitted.")
    lk_phn.add_argument("--lake-root", default=None)

    lk_pbg = lk_pools_sub.add_parser("add-background", help="Add image(s) to the backgrounds pool")
    lk_pbg.add_argument("--image", action="append", default=[], dest="images",
                        help="Relative path to image in the lake (can be repeated)")
    lk_pbg.add_argument("--task", choices=["detect", "seg"], default=None,
                        help="Force index status update for this task. Auto-detected from path if omitted.")
    lk_pbg.add_argument("--lake-root", default=None)

    # -------------------------------------------------------------------------
    # predictive  (startup-suggestion model tooling)
    # -------------------------------------------------------------------------
    pred = sub.add_parser("predictive", help="Predictive model tooling (startup suggestion models)")
    pred_sub = pred.add_subparsers(dest="predictive_cmd", required=True)

    pred_ls = pred_sub.add_parser(
        "list-artifacts",
        help="List shared MoldTrace training-row exports available for predictive training",
    )
    pred_ls.add_argument("--shared-root", default=None, help="Override ARIA shared root")
    pred_ls.add_argument("--session-id", default=None, help="Filter by MoldTrace session UUID")
    pred_ls.add_argument("--mold-id", default=None, help="Filter by mold_id")
    pred_ls.add_argument("--material-id", default=None, help="Filter by material_id")
    pred_ls.add_argument("--machine-family", default=None, help="Filter by machine family")
    pred_ls.add_argument("--limit", type=int, default=50, help="Maximum number of exports to print")
    pred_ls.add_argument("--json", action="store_true", help="Print machine-readable JSON output")

    pred_val = pred_sub.add_parser(
        "validate-dataset",
        help="Validate a training_row_v1 JSONL file produced by MoldTrace",
    )
    pred_val.add_argument(
        "--input", "-i", required=True,
        help="Path to training_rows.jsonl (training_row_v1 format)",
    )
    pred_val.add_argument(
        "--eligible-only", action="store_true",
        help="Only report on rows where eligibility.training_ready is true",
    )
    pred_val.add_argument(
        "--summary", action="store_true",
        help="Print a dataset summary (feature counts, quality stats, defect distribution)",
    )

    pred_train = pred_sub.add_parser(
        "train",
        help="Train startup-suggestion GBT models from a training_row_v1 JSONL file",
    )
    pred_train.add_argument(
        "--input", "-i", required=True,
        help="Path to training_rows.jsonl (training_row_v1/v2 format)",
    )
    pred_train.add_argument(
        "--output-dir", "-o", default=None,
        help="Directory to write trained model artifacts. Default: a new run folder under the local predictive-runs root.",
    )
    pred_train.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    pred_train.add_argument(
        "--n-estimators", type=int, default=300,
        help="Number of LightGBM boosting rounds (default: 300)",
    )
    pred_train.add_argument(
        "--learning-rate", type=float, default=0.05,
        help="LightGBM learning rate (default: 0.05)",
    )
    pred_train.add_argument(
        "--null-strategy", choices=["native_missing", "mean_impute", "zero_impute"], default="native_missing",
        help="Strategy for missing feature values during training/runtime (default: native_missing)",
    )
    pred_train.add_argument(
        "--min-feature-presence-ratio", type=float, default=0.05,
        help="Drop feature columns present in fewer than this fraction of eligible rows (default: 0.05)",
    )
    pred_train.add_argument(
        "--mold-id", default=None,
        help="Mold ID to scope this training run. Only rows with matching mold_id are used.",
    )
    pred_train.add_argument(
        "--material-id", default=None,
        help="Material ID to scope this training run. Only rows with matching material_id are used.",
    )
    pred_train.add_argument(
        "--machine-id", default=None,
        help=(
            "Machine ID (or family) to scope this training run. "
            "Only rows whose context.machine_family matches are used. "
            "Recommended when your dataset contains data from multiple machine types "
            "with different HMI schemas."
        ),
    )
    pred_train.add_argument(
        "--allow-scope-filtering", action="store_true",
        help=(
            "When scope flags are provided, drop rows outside that scope instead of failing. "
            "Use only for exploratory pooled datasets; production training should export one JSONL per scope."
        ),
    )

    pred_bundle = pred_sub.add_parser(
        "bundle",
        help="Package trained suggestion models into a deployable .sugbundle",
    )
    pred_bundle.add_argument(
        "--train-dir", required=True,
        help="Directory written by 'predictive train' containing train_result.pkl",
    )
    pred_bundle.add_argument(
        "--model-name", required=True,
        help="Bundle model name (e.g. 'startup-suggestion')",
    )
    pred_bundle.add_argument(
        "--model-version", required=True,
        help="Semantic version string (e.g. '1.0.0')",
    )
    pred_bundle.add_argument(
        "--channel", default="stable", choices=["stable", "beta"],
        help="Release channel (default: stable)",
    )
    pred_bundle.add_argument(
        "--supersedes", default=None,
        help="bundle_id of the bundle this replaces (optional)",
    )
    pred_bundle.add_argument(
        "--sugbundle", action="store_true",
        help="Also pack the bundle directory into a .sugbundle zip archive",
    )
    pred_bundle.add_argument(
        "--mold-id", default=None,
        help="Mold ID to encode in the bundle scope block (optional, recommended)",
    )
    pred_bundle.add_argument(
        "--material-id", default=None,
        help="Material ID to encode in the bundle scope block (optional, recommended)",
    )
    pred_bundle.add_argument(
        "--machine-id", default=None,
        help="Machine ID to encode in the bundle scope block (optional, recommended)",
    )
    pred_bundle.add_argument(
        "--publish", action="store_true",
        help="Immediately publish the generated suggestion bundle using role startup_suggestion",
    )
    pred_bundle.add_argument(
        "--publish-dry-run", action="store_true",
        help="Dry-run the publish step and print the resulting catalog/index metadata",
    )

    # ---- publish ----
    sp_publish = sub.add_parser("publish", help="Publish a model bundle to the S3 model catalog")
    sp_publish.add_argument("bundle_path", type=str, help="Path to bundle directory or .mpk/.zip")
    sp_publish.add_argument("--role", required=True, help="Model role (e.g., defect_detector, monitor_segmenter)")
    sp_publish.add_argument("--channel", default="stable", choices=["stable", "beta"], help="Release channel")
    sp_publish.add_argument("--compatible-layouts", nargs="*", default=None, help="HMI layouts this model supports (default: all)")
    sp_publish.add_argument("--dry-run", action="store_true", help="Print what would be done without uploading")

    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from . import cli_handlers  # lazy import to keep startup fast

    if args.cmd == "doctor":
        return cli_handlers.handle_doctor(args)

    if args.cmd == "dataset":
        subcmd = getattr(args, "dataset_cmd", None)
        handler_name = f"handle_dataset_{subcmd.replace('-', '_')}" if subcmd else None
        handler = getattr(cli_handlers, handler_name, None) if handler_name else None
        if handler:
            return handler(args)
        build_parser().parse_args([args.cmd, "--help"])
        return 2

    if args.cmd == "train":
        return cli_handlers.handle_train(args)

    if args.cmd == "export":
        return cli_handlers.handle_export(args)

    if args.cmd == "bundle":
        return cli_handlers.handle_bundle(args)

    if args.cmd == "infer":
        return cli_handlers.handle_infer(args)

    if args.cmd == "config":
        return cli_handlers.handle_config(args)

    if args.cmd == "lake":
        return cli_handlers.handle_lake(args)

    if args.cmd == "predictive":
        return cli_handlers.handle_predictive(args)

    if args.cmd == "publish":
        return cli_handlers.handle_publish(args)

    build_parser().parse_args(["--help"])
    return 2
