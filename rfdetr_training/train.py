from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from .checkpoints import load_checkpoint_weights
from .coco import ensure_minimal_test_split, validate_coco_split
from .datasets import load_metadata


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: Path  # dataset UUID folder
    task: str  # detect|seg
    size: str  # nano|small|base|medium (detect only)
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    output_dir: Optional[Path]
    pretrained: bool
    pretrain_weights: Optional[str]
    tensorboard: bool
    wandb: bool
    early_stopping: bool
    eval: bool
    num_queries: Optional[int]
    num_select: Optional[int]
    run_test: bool
    benchmark: bool
    resume: Optional[str]
    finetune_from: Optional[str]
    use_checkpoint_model: bool
    checkpoint_key: Optional[str]
    patch_inference_mode: bool
    validate_dataset: bool


def _maybe_patch_torch_inference_mode(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import torch  # type: ignore

        if hasattr(torch, "inference_mode") and hasattr(torch, "enable_grad"):
            if not hasattr(torch, "_orig_inference_mode"):
                torch._orig_inference_mode = torch.inference_mode
            torch.inference_mode = torch.enable_grad
            print("Note: patched torch.inference_mode -> torch.enable_grad for training compatibility")
    except Exception:
        pass


def _instantiate_model(task: str, size: str, num_classes: Optional[int], pretrain_weights: Optional[str]):
    task = (task or "detect").lower().strip()
    if task == "seg":
        from rfdetr import RFDETRSegPreview  # type: ignore

        kwargs: Dict[str, Any] = {}
        if pretrain_weights:
            kwargs["pretrain_weights"] = pretrain_weights
        # seg preview may not accept num_classes; keep it best-effort
        try:
            if num_classes is not None:
                kwargs["num_classes"] = num_classes
            return RFDETRSegPreview(**kwargs) if kwargs else RFDETRSegPreview()
        except TypeError:
            return RFDETRSegPreview()

    # detect
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium  # type: ignore

    cls = {"nano": RFDETRNano, "small": RFDETRSmall, "base": RFDETRBase, "medium": RFDETRMedium}.get(size)
    if cls is None:
        raise ValueError(f"Unknown model size: {size}")

    kwargs = {}
    if pretrain_weights:
        kwargs["pretrain_weights"] = pretrain_weights
    if num_classes is not None:
        kwargs["num_classes"] = int(num_classes)
    try:
        return cls(**kwargs) if kwargs else cls()
    except TypeError:
        # older constructors may not accept num_classes or pretrain_weights
        return cls()


def train(cfg: TrainConfig) -> int:
    _maybe_patch_torch_inference_mode(cfg.patch_inference_mode)

    dataset_dir = cfg.dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        print(f"Dataset folder not found: {dataset_dir}", file=sys.stderr)
        return 2

    coco_dir = dataset_dir / "coco"
    if not coco_dir.exists():
        print(f"COCO folder not found under dataset: {coco_dir}", file=sys.stderr)
        return 3

    created = ensure_minimal_test_split(coco_dir)
    if created is not None:
        print(f"Note: created minimal test split annotations at {created}")

    if cfg.validate_dataset:
        for split in ("train", "valid"):
            v = validate_coco_split(coco_dir / split, task=cfg.task, check_images_exist=False)
            for w in v.warnings:
                print("Warning:", w, file=sys.stderr)
            if not v.ok:
                for e in v.errors:
                    print("Error:", e, file=sys.stderr)
                return 4

    md = load_metadata(dataset_dir)
    class_names = md.get("class_names", []) or []
    num_classes = len(class_names) if class_names else None

    out_dir = cfg.output_dir.expanduser().resolve() if cfg.output_dir else (dataset_dir / "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.resume and cfg.finetune_from:
        print("Error: --resume and --finetune-from are mutually exclusive.", file=sys.stderr)
        return 5

    # instantiate model
    try:
        model = _instantiate_model(cfg.task, cfg.size, num_classes, cfg.pretrain_weights)
    except Exception as e:
        print("Failed to instantiate model. Is `rfdetr` installed in this environment?", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 6

    if cfg.pretrained and not cfg.pretrain_weights:
        maybe_dl = getattr(model, "maybe_download_pretrain_weights", None)
        if callable(maybe_dl):
            print("Ensuring pretrained weights are available (downloading if needed)...")
            try:
                maybe_dl()
            except Exception as e:
                print(f"Warning: automatic pretrained download failed: {e}", file=sys.stderr)

    # optional finetune-from weights
    if cfg.finetune_from:
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            res = load_checkpoint_weights(
                model,
                cfg.finetune_from,
                device,
                checkpoint_key=cfg.checkpoint_key,
                allow_replace_model=cfg.use_checkpoint_model,
                verbose=True,
            )
            if res.replacement_model is not None:
                model = res.replacement_model
            if not res.ok:
                print(f"Warning: could not load finetune-from weights: {res.message}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: finetune-from load failed: {e}", file=sys.stderr)

    # build train kwargs
    train_kwargs: Dict[str, Any] = {
        "dataset_dir": str(coco_dir),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "grad_accum_steps": int(cfg.grad_accum),
        "lr": float(cfg.lr),
        "tensorboard": bool(cfg.tensorboard),
        "wandb": bool(cfg.wandb),
        "early_stopping": bool(cfg.early_stopping),
        "eval": bool(cfg.eval),
        "output_dir": str(out_dir),
        "run_test": bool(cfg.run_test),
        "do_benchmark": bool(cfg.benchmark),
    }

    # segmentation defaults / safety
    if cfg.task == "seg":
        nq = cfg.num_queries if cfg.num_queries is not None else 200
        ns = cfg.num_select if cfg.num_select is not None else nq
        if ns > nq:
            print(f"Warning: num_select ({ns}) > num_queries ({nq}); clamping num_select to {nq}.", file=sys.stderr)
            ns = nq
        train_kwargs["num_queries"] = int(nq)
        train_kwargs["num_select"] = int(ns)
    else:
        if cfg.num_queries is not None:
            train_kwargs["num_queries"] = int(cfg.num_queries)
        if cfg.num_select is not None:
            if cfg.num_queries is not None and cfg.num_select > cfg.num_queries:
                print(
                    f"Warning: num_select ({cfg.num_select}) > num_queries ({cfg.num_queries}); clamping.",
                    file=sys.stderr,
                )
                train_kwargs["num_select"] = int(cfg.num_queries)
            else:
                train_kwargs["num_select"] = int(cfg.num_select)

    if cfg.resume:
        train_kwargs["resume"] = cfg.resume

    # save config (portable-ish; store relative-ish paths when possible)
    try:
        payload = asdict(cfg)
        payload["dataset_dir"] = str(Path(payload["dataset_dir"]).as_posix())
        payload["output_dir"] = str(out_dir.as_posix())
        (out_dir / "train_args.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    try:
        model.train(**train_kwargs)  # type: ignore[attr-defined]
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        (out_dir / "training_error_trace.txt").write_text(tb, encoding="utf-8")
        print(f"Training failed: {e}", file=sys.stderr)
        print(f"Full traceback written to: {out_dir / 'training_error_trace.txt'}", file=sys.stderr)
        return 10

    print(f"Training finished. Outputs in: {out_dir}")
    return 0

