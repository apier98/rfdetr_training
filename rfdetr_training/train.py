from __future__ import annotations

import importlib.metadata
import json
import sys
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .checkpoints import load_checkpoint_weights
from .coco import (
    align_coco_categories_to_metadata,
    ensure_minimal_test_split,
    patch_coco_categories_supercategory,
    validate_coco_split,
)
from .datasets import load_metadata
from .model_factory import instantiate_rfdetr_model
from .rfdetr_patches import patch_albumentations_empty_masks


@dataclass(frozen=True)
class TrainConfig:
    dataset_dir: Path  # dataset UUID folder
    task: str  # detect|seg
    size: str  # nano|small|base|medium (seg depends on installed rfdetr version)
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    device: Optional[str]
    num_workers: Optional[int]
    resolution: Optional[int]
    output_dir: Optional[Path]
    pretrained: bool
    pretrain_weights: Optional[str]
    tensorboard: bool
    wandb: bool
    early_stopping: bool
    eval_only: bool
    num_queries: Optional[int]
    num_select: Optional[int]
    run_test: bool
    benchmark: bool
    resume: Optional[str]
    finetune_from: Optional[str]
    use_checkpoint_model: bool
    checkpoint_key: Optional[str]
    patch_inference_mode: Optional[bool]
    validate_dataset: bool
    multi_scale: Optional[bool]
    expanded_scales: Optional[bool]
    do_random_resize_via_padding: Optional[bool]
    aug_config: Optional[Dict[str, Any]]
    no_aug: bool


def _package_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _summarize_training_outputs(out_dir: Path) -> None:
    try:
        out_dir = out_dir.expanduser().resolve()
    except Exception:
        return

    candidates = [
        "checkpoint_best_total.pth",
        "checkpoint_best_regular.pth",
        "checkpoint_best_ema.pth",
        "checkpoint.pth",
        "train_args.json",
        "results.json",
        "log.txt",
        "training_error_trace.txt",
    ]
    existing = [name for name in candidates if (out_dir / name).exists()]

    # Also show numbered checkpoints if present
    numbered = sorted(out_dir.glob("checkpoint*.pth"))
    numbered_names = [p.name for p in numbered if p.name not in existing]
    if len(numbered_names) > 10:
        numbered_names = numbered_names[:5] + ["..."] + numbered_names[-5:]

    if existing or numbered_names:
        print("Outputs:")
        for name in existing:
            print(f"- {out_dir / name}")
        if numbered_names:
            print(f"- {out_dir / 'checkpoint*.pth'} ({', '.join(numbered_names)})")


def _try_git_sha(repo_root: Path) -> Optional[str]:
    try:
        if not (repo_root / ".git").exists():
            return None
        sha = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        s = sha.decode("utf-8", errors="ignore").strip()
        return s or None
    except Exception:
        return None


def _archive_previous_error_trace(out_dir: Path) -> None:
    """Avoid confusing 'training_error_trace.txt' from a previous failed run with the current run."""
    try:
        p = out_dir / "training_error_trace.txt"
        if not p.exists():
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = out_dir / f"training_error_trace_prev_{ts}.txt"
        p.rename(dst)
    except Exception:
        return


def _write_deployment_bundle(out_dir: Path, dataset_dir: Path, cfg: TrainConfig, metadata: Dict[str, Any]) -> None:
    """Write minimal, self-contained metadata to ship the trained model directory."""
    try:
        class_names = metadata.get("class_names", []) or []
        num_classes = int(len(class_names))

        repo_root = Path(__file__).resolve().parents[1]
        sha = _try_git_sha(repo_root)

        torch_ver = None
        try:
            import torch  # type: ignore

            torch_ver = getattr(torch, "__version__", None)
        except Exception:
            torch_ver = None

        rfdetr_ver = None
        try:
            import rfdetr  # type: ignore

            rfdetr_ver = getattr(rfdetr, "__version__", None)
        except Exception:
            rfdetr_ver = None

        (out_dir / "classes.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

        model_config = {
            "format_version": 1,
            "task": cfg.task,
            "size": cfg.size,
            "num_classes": num_classes,
            "class_names": class_names,
            "resolution": cfg.resolution,
            "num_queries": cfg.num_queries,
            "num_select": cfg.num_select,
            "git_sha": sha,
            "python": sys.version.split()[0],
            "torch": torch_ver,
            "rfdetr": rfdetr_ver,
            "runtime_versions": {
                "python": sys.version.split()[0],
                "torch": torch_ver,
                "torchvision": _package_version("torchvision"),
                "rfdetr": rfdetr_ver,
                "pillow": _package_version("pillow"),
                "numpy": _package_version("numpy"),
            },
            "dataset_uuid": metadata.get("uuid"),
            "dataset_name": metadata.get("name"),
        }
        (out_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")

        # Preprocess settings are part of the "portable contract" for deployment.
        # Default policy is letterbox to preserve aspect ratio (matches typical operator UIs).
        target = int(cfg.resolution) if cfg.resolution is not None else 640
        preprocess = {
            "format_version": 1,
            "resize_policy": "letterbox",
            "target_h": target,
            "target_w": target,
            "input_color": "RGB",
            "input_layout": "NCHW",
            "input_dtype": "float32",
            "input_range": "0..1",
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "note": "PIL RGB -> float32 0..1 -> ImageNet mean/std normalization; resize uses letterbox to preserve aspect ratio.",
        }
        (out_dir / "preprocess.json").write_text(json.dumps(preprocess, indent=2), encoding="utf-8")

        postprocess = {
            "format_version": 1,
            "score_threshold_default": 0.3,
            "mask_threshold_default": 0.5,
            "mask_alpha_default": 0.45,
            "note": "Defaults used by scripts/infer_helpers.py parse_model_output().",
        }
        (out_dir / "postprocess.json").write_text(json.dumps(postprocess, indent=2), encoding="utf-8")

        # Helpful when deploying the model directory alone.
        try:
            md_path = dataset_dir / "METADATA.json"
            if md_path.exists():
                (out_dir / "dataset_metadata.json").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    except Exception:
        # bundle is best-effort; never fail training due to this
        return


def _try_write_portable_checkpoint(out_dir: Path, *, checkpoint_key: Optional[str]) -> None:
    """Best-effort: write a weights-only checkpoint next to training outputs."""
    try:
        out_dir = out_dir.expanduser().resolve()
    except Exception:
        return

    # Prefer "best" checkpoints when present.
    candidates = [
        out_dir / "checkpoint_best_total.pth",
        out_dir / "checkpoint_best_ema.pth",
        out_dir / "checkpoint_best_regular.pth",
        out_dir / "checkpoint.pth",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return

    dst = out_dir / "checkpoint_portable.pth"
    try:
        import torch  # type: ignore

        from .checkpoints import save_portable_checkpoint

        ok, msg = save_portable_checkpoint(
            src_path=str(src),
            dst_path=str(dst),
            device=torch.device("cpu"),
            checkpoint_key=checkpoint_key,
            verbose=False,
        )
        if ok:
            print(f"Note: {msg}")
        else:
            print(f"Warning: could not write portable checkpoint ({dst}): {msg}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: could not write portable checkpoint ({dst}): {e}", file=sys.stderr)


class _PatchedInferenceMode:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._torch = None
        self._orig = None

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            import torch  # type: ignore

            self._torch = torch
            if hasattr(torch, "inference_mode") and hasattr(torch, "enable_grad"):
                self._orig = torch.inference_mode
                torch.inference_mode = torch.enable_grad
                print("Note: patched torch.inference_mode -> torch.enable_grad for training compatibility")
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        try:
            if self._torch is not None and self._orig is not None:
                self._torch.inference_mode = self._orig
        except Exception:
            pass
        return False


def train(cfg: TrainConfig) -> int:
    patch_default = os.name == "nt"
    patch_enabled = patch_default if cfg.patch_inference_mode is None else bool(cfg.patch_inference_mode)

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

    # Patch COCO categories to include 'supercategory' key (some RF-DETR versions assume it exists).
    for sp in ("train", "valid", "test"):
        ann_path = coco_dir / sp / "_annotations.coco.json"
        changed, msg = patch_coco_categories_supercategory(ann_path, default="")
        if changed:
            print(f"Note: {msg}")

    md = load_metadata(dataset_dir)
    class_names = md.get("class_names", []) or []
    num_classes = len(class_names) if class_names else None

    # Keep categories stable and aligned to METADATA.json for smooth training.
    if class_names:
        for sp in ("train", "valid", "test"):
            ann_path = coco_dir / sp / "_annotations.coco.json"
            ok_align, msg_align = align_coco_categories_to_metadata(ann_path, class_names=class_names, dry_run=False)
            if ok_align:
                print(f"Note: {msg_align}")
            else:
                print(f"Warning: {msg_align}", file=sys.stderr)

    if cfg.validate_dataset:
        for split in ("train", "valid"):
            v = validate_coco_split(coco_dir / split, task=cfg.task, check_images_exist=False)
            for w in v.warnings:
                print("Warning:", w, file=sys.stderr)
            if not v.ok:
                for e in v.errors:
                    print("Error:", e, file=sys.stderr)
                if cfg.task == "seg":
                    print(
                        "Hint: you're training `--task seg` but your dataset has invalid/empty segmentation. "
                        "Use `--task detect` for bbox-only labels, or ingest/convert with polygon segmentations.",
                        file=sys.stderr,
                    )
                return 4

    out_dir = cfg.output_dir.expanduser().resolve() if cfg.output_dir else (dataset_dir / "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    _archive_previous_error_trace(out_dir)

    if cfg.resume and cfg.finetune_from:
        print("Error: --resume and --finetune-from are mutually exclusive.", file=sys.stderr)
        return 5
    if (not cfg.pretrained) and cfg.pretrain_weights:
        print("Error: --no-pretrained cannot be combined with --pretrain-weights.", file=sys.stderr)
        return 5

    # instantiate model
    try:
        # Pretraining behavior:
        # - By default, match upstream RF-DETR fine-tuning behavior by allowing the selected
        #   model class to use its default pretrained weights.
        # - --no-pretrained disables that and starts from scratch.
        # - --pretrain-weights overrides the default pretrained source with a user-provided path.
        if cfg.pretrain_weights:
            pretrain_weights = cfg.pretrain_weights
            force_kwarg = True
        elif cfg.pretrained:
            pretrain_weights = None
            force_kwarg = False
        else:
            pretrain_weights = None
            force_kwarg = True

        model, cls_name, size_applied = instantiate_rfdetr_model(
            cfg.task,
            cfg.size,
            num_classes=num_classes,
            pretrain_weights=pretrain_weights,
            force_pretrain_weights_kwarg=force_kwarg,
        )
        if cfg.task.lower().strip() == "seg" and not size_applied:
            print(f"Note: --size {cfg.size!r} not applied for segmentation; using {cls_name}.")
        else:
            print(f"Note: instantiated model: {cls_name} (task={cfg.task}, size={cfg.size}).")
    except Exception as e:
        print("Failed to instantiate model. Is `rfdetr` installed in this environment?", file=sys.stderr)
        print(str(e), file=sys.stderr)
        return 6

    if cfg.task.lower().strip() == "seg":
        try:
            if patch_albumentations_empty_masks():
                print("Note: patched RF-DETR Albumentations wrapper to allow empty-mask/background seg samples.")
        except Exception as e:
            print(f"Warning: could not apply RF-DETR empty-mask patch: {e}", file=sys.stderr)

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
        # RF-DETR's `eval` flag behaves like DETR's `--eval`: run evaluation and exit.
        "eval": bool(cfg.eval_only),
        "output_dir": str(out_dir),
        "run_test": bool(cfg.run_test),
        "do_benchmark": bool(cfg.benchmark),
    }

    if cfg.device:
        train_kwargs["device"] = str(cfg.device)

    # On Windows, default to num_workers=0 unless user overrides (common multiprocessing pain point).
    num_workers = cfg.num_workers
    if num_workers is None and os.name == "nt":
        num_workers = 0
        print("Note: Windows detected; defaulting --num-workers to 0 for stability.")
    if num_workers is not None:
        train_kwargs["num_workers"] = int(num_workers)

    if cfg.resolution is not None:
        res = int(cfg.resolution)
        if res <= 0:
            print("Error: --resolution must be > 0", file=sys.stderr)
            return 7
        if res % 32 != 0:
            print(
                f"Warning: resolution={res} is not divisible by 32. Some RF-DETR backbones require 32-divisible sizes.",
                file=sys.stderr,
            )
        if res % 224 != 0:
            # from community reports: tutorials mention 56 but some asserts require 32; LCM is 224
            print(
                f"Note: resolution={res}. If you hit resolution divisibility asserts, try 224/448/etc.",
                file=sys.stderr,
            )
        train_kwargs["resolution"] = res

    if bool(cfg.no_aug):
        train_kwargs["aug_config"] = {}
        print("Note: augmentations disabled via --no-aug (aug_config={}).")
    elif cfg.aug_config is not None:
        train_kwargs["aug_config"] = cfg.aug_config
        print("Note: using custom aug_config from CLI.")

    # multi-scale / resize policies
    if cfg.task.lower().strip() == "seg":
        # RF-DETR seg can crash with "masks cannot be empty" when multi-scale training crops away all instances.
        # Default to a stable, deployment-friendly regime unless explicitly overridden.
        if cfg.multi_scale is None:
            train_kwargs["multi_scale"] = False
            train_kwargs["expanded_scales"] = False
            print("Note: --task seg: defaulting multi_scale=False for stability (pass --multi-scale to enable).")
        else:
            train_kwargs["multi_scale"] = bool(cfg.multi_scale)
            if cfg.expanded_scales is not None:
                train_kwargs["expanded_scales"] = bool(cfg.expanded_scales)
    else:
        if cfg.multi_scale is not None:
            train_kwargs["multi_scale"] = bool(cfg.multi_scale)
        if cfg.expanded_scales is not None:
            train_kwargs["expanded_scales"] = bool(cfg.expanded_scales)
    if cfg.do_random_resize_via_padding is not None:
        train_kwargs["do_random_resize_via_padding"] = bool(cfg.do_random_resize_via_padding)

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
        with _PatchedInferenceMode(patch_enabled):
            if cfg.eval_only:
                print("Note: --eval-only enabled; running evaluation only (no training epochs).")
            model.train(**train_kwargs)  # type: ignore[attr-defined]
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        (out_dir / "training_error_trace.txt").write_text(tb, encoding="utf-8")
        msg = str(e)
        print(f"Training failed: {e}", file=sys.stderr)
        print(f"Full traceback written to: {out_dir / 'training_error_trace.txt'}", file=sys.stderr)
        _summarize_training_outputs(out_dir)

        low = msg.lower()
        if "inference tensors cannot be saved for backward" in low:
            print(
                "Hint: this is a known issue on some setups. Re-run with `--patch-inference-mode` "
                "(or leave it on auto for Windows).",
                file=sys.stderr,
            )
        if "freeze_support" in low or "attempt has been made to start a new process" in low:
            print(
                "Hint: Windows multiprocessing issue. Try `--num-workers 0` and run from a terminal (not inside a notebook).",
                file=sys.stderr,
            )
        if "masks cannot be empty" in low:
            print(
                "Hint: segmentation training crashed because a transform produced an empty mask set for a sample. "
                "Keep multi-scale/random-resize conservative, and prune invalid or tiny masks with "
                "`python -m rfdetr_training dataset prune-empty-masks -d <DATASET_DIR> --split train|valid` "
                "or `python -m rfdetr_training dataset prune-small-masks ...`.",
                file=sys.stderr,
            )
        if "out of memory" in low or "cuda out of memory" in low:
            print(
                "Hint: OOM. Reduce `--batch-size` or `--resolution`. "
                "Note: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not supported on all platforms (e.g. Windows).",
                file=sys.stderr,
            )
        return 10

    print(f"Training finished. Outputs in: {out_dir}")
    _write_deployment_bundle(out_dir, dataset_dir, cfg, md)
    _try_write_portable_checkpoint(out_dir, checkpoint_key=cfg.checkpoint_key)
    _summarize_training_outputs(out_dir)
    return 0
