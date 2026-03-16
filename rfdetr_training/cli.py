from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List

from .coco import (
    align_coco_categories_to_metadata,
    ensure_minimal_test_split,
    normalize_coco_category_ids,
    prune_empty_masks_in_split,
    prune_too_small_masks_in_split,
    reset_coco_dir,
    validate_coco_split,
)
from .coco_merge import merge_coco_into_split
from .datasets import create_dataset, load_metadata, yolo_to_coco
from .bundle import create_bundle
from .export import export_onnx, export_tensorrt_from_onnx
from .ingest import ingest_labels_inbox
from .infer import infer_from_bundle
from .train import TrainConfig, train


def _parse_classes(values: List[str] | None, classes_file: str | None) -> List[str]:
    cls: List[str] = []
    if values:
        for entry in values:
            parts = [p.strip() for p in entry.replace(",", " ").split() if p.strip()]
            cls.extend(parts)
    if classes_file:
        p = Path(classes_file).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Classes file not found: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                cls.append(name)
    # dedupe, preserve order
    seen = set()
    out: List[str] = []
    for c in cls:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _load_jsonish(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    txt = p.read_text(encoding="utf-8")
    try:
        return json.loads(txt)
    except Exception:
        # best-effort YAML support if installed
        try:
            import yaml  # type: ignore

            obj = yaml.safe_load(txt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    raise ValueError(f"Could not parse config as JSON (or YAML): {p}")


def _load_trained_model_config(dataset_dir: Path) -> dict:
    p = dataset_dir.expanduser().resolve() / "models" / "model_config.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rfdetrw", description="RF-DETR training workspace (CLI)")
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
    ex.add_argument("--format", choices=["onnx", "tensorrt"], default="onnx")
    ex.add_argument("--output", "-o", default=None, help="Output path (.onnx or .engine). Default: datasets/<UUID>/exports/")
    ex.add_argument("--device", default=None, help="Device for export (e.g. cuda:0, cpu)")
    ex.add_argument("--height", type=int, default=640)
    ex.add_argument("--width", type=int, default=640)
    ex.add_argument("--opset", type=int, default=18, help="ONNX opset")
    ex.add_argument("--dynamic", action="store_true", help="Dynamic input H/W axes (ONNX only)")
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
        choices=["onnx", "tensorrt"],
        help="Extra exported formats to include. Default behavior already ships ONNX as the primary bundle artifact.",
    )
    bd.add_argument("--opset", type=int, default=18, help="ONNX opset")
    bd.add_argument("--dynamic", action="store_true", help="ONNX: export dynamic H/W axes")
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
    bd.add_argument("--overwrite", action="store_true", help="Overwrite output dir if it exists")

    # infer (bundle)
    inf = sub.add_parser("infer", help="Run inference using a deployment bundle directory (debug/smoke check)")
    inf.add_argument("--bundle-dir", "-b", required=True)
    inf.add_argument("--image", "-i", required=True)
    inf.add_argument("--weights", "-w", default=None, help="Override fallback checkpoint path (default: <bundle>/checkpoint.pth)")
    inf.add_argument("--backend", choices=["auto", "tensorrt", "onnx", "pytorch"], default="auto", help="Inference backend. Default: auto (prefer TensorRT, then ONNX)")
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
    ds_create.add_argument("--root", "-r", default="datasets")
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
    ds_ing.add_argument("--yolo-task", choices=["detect", "seg"], default="detect", help="How to interpret YOLO labels during conversion")
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
    tr.add_argument("--finetune-from", default=None, help="Load weights (best-effort) then start a new run")
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
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "doctor":
        # keep this lightweight: just print versions and common hints.
        print("rfdetr_training doctor")
        print(f"- platform: {sys.platform}")
        print(f"- python: {sys.version.split()[0]}")
        try:
            import torch  # type: ignore

            print(f"- torch: {torch.__version__}")
            print(f"- cuda available: {torch.cuda.is_available()}")
        except Exception as e:
            print(f"- torch: not importable ({e})")

        try:
            import rfdetr  # type: ignore

            ver = getattr(rfdetr, "__version__", None)
            print(f"- rfdetr: {ver or 'importable'}")
            # Quick model surface check (helpful when debugging size/preview fallbacks).
            sizes = ["nano", "small", "base", "medium", "large", "xlarge", "2xlarge"]
            suf = {"nano": "Nano", "small": "Small", "base": "Base", "medium": "Medium", "large": "Large", "xlarge": "XLarge", "2xlarge": "2XLarge"}
            det = [s for s in sizes if hasattr(rfdetr, f"RFDETR{suf[s]}")]
            seg = [s for s in sizes if hasattr(rfdetr, f"RFDETRSeg{suf[s]}") or hasattr(rfdetr, f"RFDETR{suf[s]}Seg")]
            has_preview = hasattr(rfdetr, "RFDETRSegPreview")
            if det:
                print(f"- rfdetr detect sizes: {', '.join(det)}")
            if seg:
                print(f"- rfdetr seg sizes: {', '.join(seg)}")
            if has_preview and not seg:
                print("- rfdetr seg: only RFDETRSegPreview found (deprecated upstream)")
        except Exception as e:
            print(f"- rfdetr: not importable ({e})")

        if os.name == "nt":
            print("- hint: on Windows, set --num-workers 0 to avoid multiprocessing issues")
        print("- hint: if you see 'Inference tensors cannot be saved for backward', enable --patch-inference-mode")
        print(
            "- hint: if you see OOM, reduce --batch-size/--resolution; "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is not supported on all platforms (e.g. Windows)"
        )
        return 0

    if args.cmd == "dataset" and args.dataset_cmd == "create":
        try:
            classes = _parse_classes(args.classes, args.classes_file)
            layout = create_dataset(
                root=Path(args.root),
                uuid_str=args.uuid,
                name=args.name,
                force=bool(args.force),
                no_readme=bool(args.no_readme),
                class_names=classes,
            )
            print(f"Created dataset: {layout.dataset_dir}")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2

    if args.cmd == "dataset" and args.dataset_cmd == "yolo-to-coco":
        exts = [e.strip().lower() for e in args.images_ext.split(",") if e.strip()]
        try:
            yolo_to_coco(
                dataset_dir=Path(args.dataset_dir),
                task=args.task,
                train_ratio=float(args.train_ratio),
                seed=int(args.seed),
                copy_images=bool(args.copy_images),
                exts=exts,
                validate=bool(args.validate),
                validate_only=bool(args.validate_only),
                out_dir=(Path(args.out_dir) if args.out_dir else None),
            )
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2

    if args.cmd == "dataset" and args.dataset_cmd == "validate":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        coco_dir = dataset_dir / "coco"
        if args.split in ("all", "test"):
            created = ensure_minimal_test_split(coco_dir)
            if created is not None:
                print(f"Note: created minimal test split annotations at {created}")
        splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
        ok = True
        for sp in splits:
            v = validate_coco_split(coco_dir / sp, task=args.task, check_images_exist=bool(args.check_images))
            if v.ok:
                print(f"{sp}: OK")
            else:
                ok = False
                print(f"{sp}: FAILED", file=sys.stderr)
            for w in v.warnings:
                print(f"{sp}: Warning: {w}", file=sys.stderr)
            for e in v.errors:
                print(f"{sp}: Error: {e}", file=sys.stderr)
        return 0 if ok else 3

    if args.cmd == "dataset" and args.dataset_cmd == "prune-empty-masks":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        coco_dir = dataset_dir / "coco"
        splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
        ok = True
        for sp in splits:
            res = prune_empty_masks_in_split(coco_dir / sp, dry_run=bool(args.dry_run))
            if not res.ok:
                ok = False
                print(f"{sp}: Error: {res.message}", file=sys.stderr)
                continue
            print(res.message)
            if res.backup_path is not None:
                print(f"{sp}: backup: {res.backup_path}")
        return 0 if ok else 3

    if args.cmd == "dataset" and args.dataset_cmd == "prune-small-masks":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        coco_dir = dataset_dir / "coco"
        splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
        ok = True
        for sp in splits:
            res = prune_too_small_masks_in_split(
                coco_dir / sp,
                resolution=int(args.resolution),
                min_scaled_area=float(args.min_scaled_area),
                dry_run=bool(args.dry_run),
            )
            if not res.ok:
                ok = False
                print(f"{sp}: Error: {res.message}", file=sys.stderr)
                continue
            print(res.message)
            if res.backup_path is not None:
                print(f"{sp}: backup: {res.backup_path}")
        return 0 if ok else 3

    if args.cmd == "dataset" and args.dataset_cmd == "normalize-coco-ids":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        coco_dir = dataset_dir / "coco"
        splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
        ok = True
        for sp in splits:
            ann_path = coco_dir / sp / "_annotations.coco.json"
            success, msg = normalize_coco_category_ids(ann_path, dry_run=bool(args.dry_run))
            if success:
                print(msg)
            else:
                ok = False
                print(f"{sp}: {msg}", file=sys.stderr)

        # best-effort sync METADATA.json class_names from normalized categories (train split)
        if ok and not args.dry_run:
            md_path = dataset_dir / "METADATA.json"
            if md_path.exists():
                try:
                    import json

                    train_ann = coco_dir / "train" / "_annotations.coco.json"
                    cats = json.loads(train_ann.read_text(encoding="utf-8")).get("categories", []) or []
                    cats = sorted(cats, key=lambda c: int(c.get("id", 0)))
                    class_names = [str(c.get("name", c.get("id"))) for c in cats]
                    md = load_metadata(dataset_dir)
                    md["class_names"] = class_names
                    md_path.write_text(json.dumps(md, indent=2), encoding="utf-8")
                    print(f"Updated METADATA.json class_names from COCO categories: {md_path}")
                except Exception as e:
                    print(f"Warning: could not update METADATA.json: {e}", file=sys.stderr)

        return 0 if ok else 3

    if args.cmd == "dataset" and args.dataset_cmd == "align-metadata":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        coco_dir = dataset_dir / "coco"
        md = load_metadata(dataset_dir)
        class_names = md.get("class_names", []) or []
        if not class_names:
            print(f"Error: METADATA.json has empty class_names: {dataset_dir / 'METADATA.json'}", file=sys.stderr)
            return 2

        splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
        ok = True
        for sp in splits:
            ann_path = coco_dir / sp / "_annotations.coco.json"
            success, msg = align_coco_categories_to_metadata(ann_path, class_names=class_names, dry_run=bool(args.dry_run))
            if success:
                print(msg)
            else:
                ok = False
                print(f"{sp}: {msg}", file=sys.stderr)
        return 0 if ok else 3

    if args.cmd == "dataset" and args.dataset_cmd == "reset-coco":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        ok, msg = reset_coco_dir(dataset_dir, backup=(not bool(args.no_backup)))
        if not ok:
            print(f"Error: {msg}", file=sys.stderr)
            return 2
        print(msg)
        return 0

    if args.cmd == "dataset" and args.dataset_cmd == "import-coco":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        src_json = Path(args.coco_json)
        src_images = Path(args.images_dir) if args.images_dir else None

        metadata_map = None
        if args.align_metadata:
            md = load_metadata(dataset_dir)
            class_names = md.get("class_names", []) or []
            metadata_map = {str(name): idx for idx, name in enumerate(class_names)}

        res = merge_coco_into_split(
            dataset_dir=dataset_dir,
            split=args.split,
            src_json=src_json,
            src_images_dir=src_images,
            mode=args.mode,
            rename=bool(args.rename),
            pad=int(args.pad),
            metadata_map=metadata_map,
            dry_run=bool(args.dry_run),
        )
        if not res.ok:
            print(f"Error: {res.message}", file=sys.stderr)
            return 2
        print(res.message)

        # keep categories aligned and stable if requested
        if args.align_metadata:
            md = load_metadata(dataset_dir)
            class_names = md.get("class_names", []) or []
            if class_names:
                ann_path = dataset_dir / "coco" / args.split / "_annotations.coco.json"
                ok2, msg2 = align_coco_categories_to_metadata(ann_path, class_names=class_names, dry_run=False)
                if not ok2:
                    print(f"Warning: {msg2}", file=sys.stderr)
                else:
                    print(f"Note: {msg2}")

        return 0

    if args.cmd == "dataset" and args.dataset_cmd == "ingest":
        exts = [e.strip().lower() for e in args.images_ext.split(",") if e.strip()]
        if str(args.yolo_task).strip().lower() == "seg" and bool(args.include_background):
            print(
                "Note: include-background is enabled. Segmentation training can crash if an image has 0 masks "
                "('masks cannot be empty'). Consider `--no-include-background` for seg datasets.",
                file=sys.stderr,
            )
        res = ingest_labels_inbox(
            dataset_dir=Path(args.dataset_dir),
            train_ratio=float(args.train_ratio),
            seed=int(args.seed),
            yolo_task=args.yolo_task,
            images_ext=exts,
            mode=args.mode,
            align_metadata=bool(args.align_metadata),
            include_background=bool(args.include_background),
            dry_run=bool(args.dry_run),
        )
        if not res.ok:
            print(f"Error: {res.message}", file=sys.stderr)
            return 2
        print(res.message)
        return 0

    if args.cmd == "train":
        aug_cfg = None
        if getattr(args, "aug_config", None):
            aug_cfg = _load_jsonish(str(args.aug_config))
        cfg = TrainConfig(
            dataset_dir=Path(args.dataset_dir),
            task=args.task,
            size=args.size,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            grad_accum=int(args.grad_accum),
            lr=float(args.lr),
            device=args.device,
            num_workers=args.num_workers,
            resolution=args.resolution,
            output_dir=(Path(args.output_dir) if args.output_dir else None),
            pretrained=bool(args.pretrained),
            pretrain_weights=args.pretrain_weights,
            tensorboard=bool(args.tensorboard),
            wandb=bool(args.wandb),
            early_stopping=bool(args.early_stopping),
            eval_only=bool(getattr(args, "eval_only", False)),
            num_queries=args.num_queries,
            num_select=args.num_select,
            run_test=bool(args.run_test),
            benchmark=bool(args.benchmark),
            resume=args.resume,
            finetune_from=args.finetune_from,
            use_checkpoint_model=bool(args.use_checkpoint_model),
            checkpoint_key=args.checkpoint_key,
            patch_inference_mode=args.patch_inference_mode,
            validate_dataset=(not bool(args.no_validate)),
            multi_scale=getattr(args, "multi_scale", None),
            expanded_scales=getattr(args, "expanded_scales", None),
            do_random_resize_via_padding=getattr(args, "do_random_resize_via_padding", None),
            aug_config=aug_cfg,
            no_aug=bool(getattr(args, "no_aug", False)),
        )
        return int(train(cfg))

    if args.cmd == "export":
        dataset_dir = Path(args.dataset_dir)
        weights = Path(args.weights)
        out = Path(args.output) if args.output else None
        trained_model_cfg = _load_trained_model_config(dataset_dir)
        task = str(args.task or trained_model_cfg.get("task") or "detect").strip().lower()
        size = str(args.size or trained_model_cfg.get("size") or "nano").strip().lower()

        if args.format == "onnx":
            res = export_onnx(
                dataset_dir=dataset_dir,
                weights=weights,
                task=task,
                size=size,
                output=out,
                device=args.device,
                height=int(args.height),
                width=int(args.width),
                opset=int(args.opset),
                dynamic=bool(args.dynamic),
                use_checkpoint_model=bool(args.use_checkpoint_model),
                checkpoint_key=args.checkpoint_key,
                strict=bool(args.strict),
            )
            if not res.ok:
                print(f"Error: {res.message}", file=sys.stderr)
                return 2
            print(res.message)
            return 0

        # tensorrt: export ONNX first (static by design), then build engine via trtexec
        tmp_onnx = None
        if out is not None and out.suffix.lower() == ".onnx":
            tmp_onnx = out
            engine_out = out.with_suffix(".engine")
        elif out is not None and out.suffix.lower() == ".engine":
            engine_out = out
            tmp_onnx = out.with_suffix(".onnx")
        else:
            engine_out = out

        onnx_res = export_onnx(
            dataset_dir=dataset_dir,
            weights=weights,
            task=task,
            size=size,
            output=tmp_onnx,
            device=args.device,
            height=int(args.height),
            width=int(args.width),
            opset=int(args.opset),
            dynamic=False,
            use_checkpoint_model=bool(args.use_checkpoint_model),
            checkpoint_key=args.checkpoint_key,
            strict=bool(args.strict),
            batchless_input=True,
        )
        if not onnx_res.ok or onnx_res.output_path is None:
            print(f"Error: {onnx_res.message}", file=sys.stderr)
            return 2

        trt_res = export_tensorrt_from_onnx(
            onnx_path=onnx_res.output_path,
            engine_path=engine_out,
            height=int(args.height),
            width=int(args.width),
            fp16=bool(args.fp16),
            workspace_mb=int(args.workspace_mb),
        )
        if not trt_res.ok:
            print(f"Error: {trt_res.message}", file=sys.stderr)
            return 2
        print(trt_res.message)
        return 0

    if args.cmd == "bundle":
        dataset_dir = Path(args.dataset_dir)
        weights = Path(args.weights)
        out_dir = Path(args.output_dir) if args.output_dir else None
        exports = args.export or []
        res = create_bundle(
            dataset_dir=dataset_dir,
            weights=weights,
            task=args.task,
            size=args.size,
            output_dir=out_dir,
            height=args.height,
            width=args.width,
            exports=exports,
            device=args.device,
            opset=int(args.opset),
            dynamic_onnx=bool(args.dynamic),
            use_checkpoint_model=bool(args.use_checkpoint_model),
            checkpoint_key=args.checkpoint_key,
            strict=bool(args.strict),
            fp16=bool(args.fp16),
            workspace_mb=int(args.workspace_mb),
            portable_checkpoint=bool(args.portable_checkpoint),
            allow_raw_checkpoint_fallback=bool(args.allow_raw_checkpoint_fallback),
            include_raw_checkpoint=bool(args.include_raw_checkpoint),
            make_zip=bool(args.zip),
            overwrite=bool(args.overwrite),
        )
        if not res.ok:
            print(f"Error: {res.message}", file=sys.stderr)
            return 2
        print(res.message)
        return 0

    if args.cmd == "infer":
        bundle_dir = Path(args.bundle_dir)
        image_path = Path(args.image)
        weights = Path(args.weights) if args.weights else None
        res = infer_from_bundle(
            bundle_dir=bundle_dir,
            image_path=image_path,
            weights_path=weights,
            device=args.device,
            score_thresh=args.threshold,
            mask_thresh=args.mask_thresh,
            checkpoint_key=args.checkpoint_key,
            use_checkpoint_model=bool(args.use_checkpoint_model),
            strict=bool(args.strict),
            backend=str(args.backend),
        )
        if not res.ok or res.payload is None:
            print(f"Error: {res.message}", file=sys.stderr)
            return 2

        if args.out_image:
            try:
                from PIL import Image, ImageDraw  # type: ignore
            except Exception as e:
                print(f"Warning: PIL not available; cannot write --out-image ({e})", file=sys.stderr)
            else:
                try:
                    img = Image.open(str(image_path)).convert("RGB")

                    # masks (best-effort; requires numpy for fast conversion)
                    alpha = float(args.mask_alpha) if args.mask_alpha is not None else 0.45
                    if res.masks:
                        try:
                            import numpy as np

                            rgba = img.convert("RGBA")
                            overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
                            w_img, h_img = rgba.size
                            for i, m in enumerate(res.masks):
                                if m is None:
                                    continue
                                mm = np.asarray(m)
                                if mm.ndim != 2:
                                    continue
                                if mm.shape[0] != h_img or mm.shape[1] != w_img:
                                    mi = Image.fromarray((mm.astype(np.uint8) * 255))
                                    mi = mi.resize((w_img, h_img), resample=Image.NEAREST)
                                    mm = (np.asarray(mi) > 127)
                                r = (37 * (i + 1)) % 255
                                g = (17 * (i + 1)) % 255
                                b = (97 * (i + 1)) % 255
                                mask_img = Image.fromarray((mm.astype(np.uint8) * int(round(alpha * 255))))
                                overlay.paste((int(r), int(g), int(b), int(round(alpha * 255))), (0, 0), mask_img)
                            img = Image.alpha_composite(rgba, overlay).convert("RGB")
                        except Exception:
                            pass

                    draw = ImageDraw.Draw(img)
                    if res.boxes and res.scores and res.labels:
                        for b, sc, lab in zip(res.boxes, res.scores, res.labels):
                            x1, y1, x2, y2 = [float(x) for x in b]
                            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                            draw.text((x1 + 2, max(0, y1 - 12)), f"{int(lab)}:{float(sc):.2f}", fill=(0, 255, 0))

                    out_img = Path(args.out_image).expanduser().resolve()
                    out_img.parent.mkdir(parents=True, exist_ok=True)
                    img.save(str(out_img))
                    print(f"Wrote: {out_img}")
                except Exception as e:
                    print(f"Warning: failed to write --out-image ({e})", file=sys.stderr)

        if args.out_json:
            outp = Path(args.out_json).expanduser().resolve()
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(res.payload, indent=2), encoding="utf-8")
            print(f"Wrote: {outp}")
        else:
            print(json.dumps(res.payload, indent=2))
        return 0

    print("Unknown command", file=sys.stderr)
    return 2
