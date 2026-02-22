#!/usr/bin/env python3
"""Simple train wrapper for RF-DETR using the project's UUID layout.

Supports:
- Detection: RFDETRNano/Small/Base/Medium
- Segmentation: RFDETRSegPreview

Fixes / safeguards:
1) Some RF-DETR seg configs can default to num_select > num_queries (topk crash).
   We auto-set/clamp num_select <= num_queries for segmentation unless user overrides.

2) RF-DETR training may try to run benchmark / test (run_test=True / do_benchmark=True)
   and crash if coco/test is empty (common when you don't have a real test split).
   We default run_test=False and do_benchmark=False, and we auto-disable run_test if test has no images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Workaround: some RF-DETR codepaths may use torch.inference_mode() in training paths.
try:
    import torch  # type: ignore

    if hasattr(torch, "inference_mode") and hasattr(torch, "enable_grad"):
        if not hasattr(torch, "_orig_inference_mode"):
            torch._orig_inference_mode = torch.inference_mode
        torch.inference_mode = torch.enable_grad
        print("Note: patched torch.inference_mode to avoid inference-tensor errors during training")
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RF-DETR model wrapper")

    p.add_argument("--dataset-dir", "-d", required=True, help="Path to dataset UUID folder (contains coco/)")
    p.add_argument(
        "--task",
        choices=["detect", "seg"],
        default="detect",
        help="Task to train: detect (default) or seg (segmentation)",
    )

    # Detection-only (ignored for segmentation)
    p.add_argument(
        "--size",
        choices=["nano", "small", "base", "medium"],
        default="nano",
        help="Detection model size to train (ignored for --task seg). Default: nano",
    )

    p.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    p.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--output-dir", "-o", help="Output directory for checkpoints (defaults to datasets/<UUID>/models)")
    p.add_argument("--resume", help="Path to checkpoint.pth to resume training from")

    p.add_argument("--pretrained", action="store_true", help="Use pretrained weights (download if needed)")
    p.add_argument("--pretrain-weights", help="Path to a local pretrain weights file to load instead of downloading")

    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--early-stopping", action="store_true", help="Enable early stopping based on validation mAP")
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval/validation after each epoch (workaround for metric calculation errors)",
    )

    # Advanced / overrides (useful to fix upstream defaults)
    p.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Override num_queries (if supported by rfdetr train()).",
    )
    p.add_argument(
        "--num-select",
        type=int,
        default=None,
        help="Override num_select (if supported by rfdetr train()). Important: must be <= num_queries.",
    )

    # Benchmark/test controls (often crash if coco/test is empty)
    p.add_argument(
        "--run-test",
        action="store_true",
        help="Run test at end (requires non-empty coco/test). Default: disabled.",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark (can be slow / can fail on some setups). Default: disabled.",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        print(f"Dataset folder not found: {dataset_dir}", file=sys.stderr)
        return 2

    coco_path = dataset_dir / "coco"
    if not coco_path.exists():
        print(f"COCO folder not found under dataset. Run yolo_to_coco first: {coco_path}", file=sys.stderr)
        return 3

    # Choose model class
    try:
        if args.task == "seg":
            if args.size != "nano":
                print("Warning: --size is ignored for --task seg (segmentation).", file=sys.stderr)
            from rfdetr import RFDETRSegPreview as ModelClass  # type: ignore
        else:
            if args.size == "nano":
                from rfdetr import RFDETRNano as ModelClass  # type: ignore
            elif args.size == "small":
                from rfdetr import RFDETRSmall as ModelClass  # type: ignore
            elif args.size == "base":
                from rfdetr import RFDETRBase as ModelClass  # type: ignore
            elif args.size == "medium":
                from rfdetr import RFDETRMedium as ModelClass  # type: ignore
            else:
                raise ImportError("Unsupported model size")
    except ImportError as e:
        print(
            "Failed to import RF-DETR model classes. Ensure `rfdetr` is installed in the environment.",
            file=sys.stderr,
        )
        print(str(e), file=sys.stderr)
        return 4

    # Instantiate model
    model_kwargs = {}
    if args.pretrain_weights:
        model_kwargs["pretrain_weights"] = args.pretrain_weights
    model = ModelClass(**model_kwargs) if model_kwargs else ModelClass()

    # If user requested pretrained weights but did not provide a path, attempt download
    if args.pretrained and not args.pretrain_weights:
        maybe_dl = getattr(model, "maybe_download_pretrain_weights", None)
        if callable(maybe_dl):
            print("Ensuring pretrained weights are available (downloading if needed)...")
            try:
                maybe_dl()
            except Exception as e:
                print(f"Warning: automatic pretrained download failed: {e}", file=sys.stderr)

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (dataset_dir / "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build train kwargs
    train_kwargs = {
        "dataset_dir": str(coco_path),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum),
        "lr": float(args.lr),
        "tensorboard": bool(args.tensorboard),
        "wandb": bool(args.wandb),
        "early_stopping": bool(args.early_stopping),
        "eval": (not args.skip_eval),
        "output_dir": str(out_dir),

        # IMPORTANT: default to False to avoid crashes on empty test split
        "run_test": bool(args.run_test),
        "do_benchmark": bool(args.benchmark),
    }
    if args.resume:
        train_kwargs["resume"] = args.resume

    # ---- Fix: num_select must be <= num_queries (topk out of range)
    # If user provided overrides, use them (but enforce constraint).
    # If not provided and task=seg, default to num_queries=200, num_select=200 as a safe pair.
    if args.task == "seg":
        nq = args.num_queries if args.num_queries is not None else 200
        ns = args.num_select if args.num_select is not None else nq
        if ns > nq:
            print(f"Warning: --num-select ({ns}) > --num-queries ({nq}). Clamping num_select to {nq}.", file=sys.stderr)
            ns = nq
        train_kwargs["num_queries"] = int(nq)
        train_kwargs["num_select"] = int(ns)
    else:
        if args.num_queries is not None:
            train_kwargs["num_queries"] = int(args.num_queries)
        if args.num_select is not None:
            if args.num_queries is not None and args.num_select > args.num_queries:
                print(
                    f"Warning: --num-select ({args.num_select}) > --num-queries ({args.num_queries}). "
                    f"Clamping num_select to {args.num_queries}.",
                    file=sys.stderr,
                )
                train_kwargs["num_select"] = int(args.num_queries)
            else:
                train_kwargs["num_select"] = int(args.num_select)

    # Ensure test split exists (RF-DETR training expects train/valid/test folders)
    test_ann = coco_path / "test" / "_annotations.coco.json"
    if not test_ann.exists():
        src_ann = None
        for candidate in [
            coco_path / "valid" / "_annotations.coco.json",
            coco_path / "train" / "_annotations.coco.json",
        ]:
            if candidate.exists():
                src_ann = candidate
                break

        if src_ann is not None:
            try:
                src = json.loads(src_ann.read_text(encoding="utf-8"))
                categories = src.get("categories", [])
            except Exception:
                categories = []
        else:
            categories = []

        test_dir = coco_path / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        minimal = {
            "info": {"description": "auto-generated empty test split", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        (test_dir / "_annotations.coco.json").write_text(json.dumps(minimal, indent=2), encoding="utf-8")
        print(f"Note: created minimal test annotations at {test_dir / '_annotations.coco.json'}")

    # Auto-disable run_test if test split is empty (prevents list index out of range in benchmark/test)
    try:
        test_json = json.loads((coco_path / "test" / "_annotations.coco.json").read_text(encoding="utf-8"))
        if not test_json.get("images"):
            if train_kwargs.get("run_test"):
                print("Warning: coco/test is empty; disabling run_test to avoid benchmark crash.", file=sys.stderr)
            train_kwargs["run_test"] = False
    except Exception:
        train_kwargs["run_test"] = False

    # Save train configuration
    (out_dir / "train_args.json").write_text(json.dumps(train_kwargs, indent=2), encoding="utf-8")

    # Logs
    if args.task == "seg":
        print(
            f"Starting training: task=seg, epochs={args.epochs}, batch_size={args.batch_size}, grad_accum={args.grad_accum}"
        )
        print(f"Using num_queries={train_kwargs.get('num_queries')} num_select={train_kwargs.get('num_select')}")
    else:
        print(
            f"Starting training: task=detect, size={args.size}, epochs={args.epochs}, "
            f"batch_size={args.batch_size}, grad_accum={args.grad_accum}"
        )
        if "num_queries" in train_kwargs or "num_select" in train_kwargs:
            print(f"Overrides: num_queries={train_kwargs.get('num_queries')} num_select={train_kwargs.get('num_select')}")

    print(f"Dataset coco dir: {coco_path}")
    print(f"Output dir: {out_dir}")

    if train_kwargs.get("run_test"):
        print("run_test: enabled")
    else:
        print("run_test: disabled")

    if train_kwargs.get("do_benchmark"):
        print("do_benchmark: enabled")
    else:
        print("do_benchmark: disabled")

    # If user provided a resume checkpoint, attempt a selective load of matching parameter
    # keys (same name and same shape) to avoid errors when checkpoint and current model
    # architectures differ (e.g., different num_queries, segmentation heads, patch sizes).
    if args.resume:
        try:
            import torch as _torch

            # Attempt to load checkpoint; newer torch versions restrict allowed globals by default
            # which can raise a WeightsUnpickler error. Try a safe allowlist context first,
            # then fall back to loading with weights_only=False if necessary (trusted files only).
            ck = None
            try:
                ck = _torch.load(args.resume, map_location="cpu")
            except Exception as e_load:
                # Try to allow argparse.Namespace if available
                try:
                    import argparse as _argparse

                    if hasattr(_torch, "serialization"):
                        ser = _torch.serialization
                        if hasattr(ser, "add_safe_globals"):
                            with ser.add_safe_globals([_argparse.Namespace]):
                                ck = _torch.load(args.resume, map_location="cpu")
                        elif hasattr(ser, "safe_globals"):
                            with ser.safe_globals([_argparse.Namespace]):
                                ck = _torch.load(args.resume, map_location="cpu")
                        else:
                            # Last resort: load with weights_only=False (only if you trust the file)
                            ck = _torch.load(args.resume, map_location="cpu", weights_only=False)
                    else:
                        ck = _torch.load(args.resume, map_location="cpu", weights_only=False)
                except Exception:
                    # Final fallback: try weights_only=False without context
                    ck = _torch.load(args.resume, map_location="cpu", weights_only=False)

            # checkpoint containers vary: try common keys
            ck_state = None
            if isinstance(ck, dict):
                for cand in ("model", "state_dict", "state_dict_ema", "state"):
                    if cand in ck:
                        ck_state = ck[cand]
                        break
                if ck_state is None:
                    ck_state = ck
            else:
                ck_state = ck

            if isinstance(ck_state, dict):
                model_state = model.state_dict()
                filtered = {}
                skipped = []
                for k, v in ck_state.items():
                    try:
                        v_size = tuple(v.size()) if hasattr(v, "size") else None
                    except Exception:
                        v_size = None
                    m_val = model_state.get(k)
                    m_size = tuple(m_val.size()) if (m_val is not None and hasattr(m_val, "size")) else None
                    if m_val is not None and v_size is not None and m_size == v_size:
                        filtered[k] = v
                    else:
                        skipped.append(k)

                if filtered:
                    model.load_state_dict(filtered, strict=False)
                    (out_dir / "resume_filtered_loaded.txt").write_text(
                        json.dumps({"loaded_keys": len(filtered), "skipped_keys": len(skipped), "sample_skipped": skipped[:50]}, indent=2),
                        encoding="utf-8",
                    )
                    print(f"Loaded {len(filtered)} matching keys from checkpoint; skipped {len(skipped)} keys.")
                else:
                    print("No matching parameter shapes found in checkpoint; skipping model weight load.", file=sys.stderr)
            else:
                print("Checkpoint does not contain a state_dict-like mapping; skipping selective load.", file=sys.stderr)

        except Exception as e:
            print(f"Warning: selective checkpoint load failed: {e}", file=sys.stderr)
        finally:
            # Always remove resume from train kwargs to prevent the trainer from re-loading
            # the incompatible checkpoint (optimizer/epoch states) after our selective load.
            train_kwargs.pop("resume", None)

    try:
        model.train(**train_kwargs)
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        (out_dir / "training_error_trace.txt").write_text(tb, encoding="utf-8")
        print(f"Training failed: {e}", file=sys.stderr)
        print(f"Full traceback written to: {out_dir / 'training_error_trace.txt'}", file=sys.stderr)

        if "selected index k out of range" in str(e).lower():
            print(
                "Hint: this usually means num_select > num_queries (topk out of range). "
                "Try: --num-queries 200 --num-select 200 (or smaller).",
                file=sys.stderr,
            )
        if "list index out of range" in str(e).lower():
            print(
                "Hint: this often happens during benchmark/test when coco/test is empty. "
                "Ensure run_test is disabled (default) or populate coco/test with images+annotations.",
                file=sys.stderr,
            )
        return 5

    print("Training finished (check output dir for checkpoints).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
