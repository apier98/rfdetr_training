from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from .coco import validate_coco_split
from .datasets import create_dataset, yolo_to_coco
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rfdetrw", description="RF-DETR training workspace (CLI)")
    sub = p.add_subparsers(dest="cmd", required=True)

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
    ds_y2c.add_argument("--validate", action="store_true")
    ds_y2c.add_argument("--validate-only", action="store_true")

    ds_val = ds_sub.add_parser("validate", help="Validate COCO split files")
    ds_val.add_argument("--dataset-dir", "-d", required=True)
    ds_val.add_argument("--task", choices=["detect", "seg"], default="detect")
    ds_val.add_argument("--split", choices=["train", "valid", "test", "all"], default="all")
    ds_val.add_argument("--check-images", action="store_true", help="Warn if image files are missing from split dir")

    # train
    tr = sub.add_parser("train", help="Train an RF-DETR model")
    tr.add_argument("--dataset-dir", "-d", required=True)
    tr.add_argument("--task", choices=["detect", "seg"], default="detect")
    tr.add_argument("--size", choices=["nano", "small", "base", "medium"], default="nano")
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--batch-size", type=int, default=4)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--output-dir", "-o", default=None)
    tr.add_argument("--pretrained", action="store_true")
    tr.add_argument("--pretrain-weights", default=None)
    tr.add_argument("--tensorboard", action="store_true")
    tr.add_argument("--wandb", action="store_true")
    tr.add_argument("--early-stopping", action="store_true")
    tr.add_argument("--skip-eval", action="store_true")
    tr.add_argument("--num-queries", type=int, default=None)
    tr.add_argument("--num-select", type=int, default=None)
    tr.add_argument("--run-test", action="store_true")
    tr.add_argument("--benchmark", action="store_true")

    # safer replacements than the old "resume but partially load and then remove resume"
    tr.add_argument("--resume", default=None, help="Resume training (trainer checkpoint)")
    tr.add_argument("--finetune-from", default=None, help="Load weights (best-effort) then start a new run")
    tr.add_argument("--use-checkpoint-model", action="store_true", help="If checkpoint contains a pickled model, use it (trusted only)")
    tr.add_argument("--checkpoint-key", default=None, help="Explicit key inside checkpoint that contains state_dict")

    tr.add_argument("--patch-inference-mode", action="store_true", help="Workaround: patch torch.inference_mode -> enable_grad")
    tr.add_argument("--no-validate", action="store_true", help="Skip dataset validation checks")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

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
            )
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2

    if args.cmd == "dataset" and args.dataset_cmd == "validate":
        dataset_dir = Path(args.dataset_dir).expanduser().resolve()
        coco_dir = dataset_dir / "coco"
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

    if args.cmd == "train":
        cfg = TrainConfig(
            dataset_dir=Path(args.dataset_dir),
            task=args.task,
            size=args.size,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            grad_accum=int(args.grad_accum),
            lr=float(args.lr),
            output_dir=(Path(args.output_dir) if args.output_dir else None),
            pretrained=bool(args.pretrained),
            pretrain_weights=args.pretrain_weights,
            tensorboard=bool(args.tensorboard),
            wandb=bool(args.wandb),
            early_stopping=bool(args.early_stopping),
            eval=(not bool(args.skip_eval)),
            num_queries=args.num_queries,
            num_select=args.num_select,
            run_test=bool(args.run_test),
            benchmark=bool(args.benchmark),
            resume=args.resume,
            finetune_from=args.finetune_from,
            use_checkpoint_model=bool(args.use_checkpoint_model),
            checkpoint_key=args.checkpoint_key,
            patch_inference_mode=bool(args.patch_inference_mode),
            validate_dataset=(not bool(args.no_validate)),
        )
        return int(train(cfg))

    print("Unknown command", file=sys.stderr)
    return 2

