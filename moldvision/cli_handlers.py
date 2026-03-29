"""Business-logic handlers for each CLI command.

Each function takes the parsed ``argparse.Namespace`` and returns an integer
exit code (0 = success, 1 = interrupted, 2 = error, 3 = validation warning).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

from . import appconfig
from .coco import (
    align_coco_categories_to_metadata,
    ensure_minimal_test_split,
    normalize_coco_category_ids,
    prune_empty_masks_in_split,
    prune_too_small_masks_in_split,
    reset_coco_dir,
    subsample_coco_split,
    validate_coco_split,
)
from .coco_merge import merge_coco_into_split
from .datasets import create_dataset, load_metadata, yolo_to_coco
from .bundle import create_bundle
from .export import export_onnx, export_tensorrt_from_onnx
from .ingest import ingest_labels_inbox
from .infer import infer_from_bundle
from .train import TrainConfig, train
from .videos import extract_frames
from .pathutil import resolve_path


def _parse_classes(values: List[str] | None, classes_file: str | None) -> List[str]:
    cls: List[str] = []
    if values:
        for entry in values:
            parts = [p.strip() for p in entry.replace(",", " ").split() if p.strip()]
            cls.extend(parts)
    if classes_file:
        p = resolve_path(classes_file)
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
    p = resolve_path(path)
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
    p = resolve_path(dataset_dir) / "models" / "model_config.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def handle_doctor(args) -> int:
    # keep this lightweight: just print versions and common hints.
    print("moldvision doctor")
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

    print(f"- config file: {appconfig.config_path()}")
    print(f"- default dataset root: {appconfig.get_default_dataset_root()}")
    print(f"- default num_workers:  {appconfig.get_default_num_workers()}")
    print(f"- inference backend:    {appconfig.get_default_inference_backend()}")
    print(f"- export format:        {appconfig.get_default_export_format()}")
    print(f"  (use 'moldvision config set <key> <value>' to change)")

    if os.name == "nt":
        print("- hint: on Windows, num-workers defaults to 0 to avoid multiprocessing issues")
    print("- hint: if you see 'Inference tensors cannot be saved for backward', enable --patch-inference-mode")
    print(
        "- hint: if you see OOM, reduce --batch-size/--resolution; "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is not supported on all platforms (e.g. Windows)"
    )
    return 0


def handle_dataset_create(args) -> int:
    try:
        classes = _parse_classes(args.classes, args.classes_file)
        root = Path(args.root) if args.root else Path(appconfig.get_default_dataset_root())
        layout = create_dataset(
            root=root,
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


def handle_dataset_yolo_to_coco(args) -> int:
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


def handle_dataset_validate(args) -> int:
    dataset_dir = resolve_path(args.dataset_dir)
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


def handle_dataset_prune_empty_masks(args) -> int:
    dataset_dir = resolve_path(args.dataset_dir)
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


def handle_dataset_prune_small_masks(args) -> int:
    dataset_dir = resolve_path(args.dataset_dir)
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


def handle_dataset_normalize_coco_ids(args) -> int:
    dataset_dir = resolve_path(args.dataset_dir)
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


def handle_dataset_align_metadata(args) -> int:
    dataset_dir = resolve_path(args.dataset_dir)
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


def handle_dataset_reset_coco(args) -> int:
    dataset_dir = resolve_path(args.dataset_dir)
    ok, msg = reset_coco_dir(dataset_dir, backup=(not bool(args.no_backup)))
    if not ok:
        print(f"Error: {msg}", file=sys.stderr)
        return 2
    print(msg)
    return 0


def handle_dataset_subsample(args) -> int:
    dataset_dir = resolve_path(args.dataset_dir)
    coco_dir = dataset_dir / "coco"
    res = subsample_coco_split(
        coco_dir / args.split,
        fraction=args.fraction,
        max_images=args.max_images,
        min_instances_per_class=args.min_instances,
        seed=args.seed,
        dry_run=bool(args.dry_run),
    )
    if not res.ok:
        print(f"Error: {res.message}", file=sys.stderr)
        return 2
    print(res.message)
    if res.backup_path is not None:
        print(f"Backup saved to: {res.backup_path}")
    return 0


def handle_dataset_import_coco(args) -> int:
    dataset_dir = resolve_path(args.dataset_dir)
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


def handle_dataset_ingest(args) -> int:
    exts = [e.strip().lower() for e in args.images_ext.split(",") if e.strip()]
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


def handle_dataset_extract_frames(args) -> int:
    out_path = None
    if args.out_dir:
        out_path = Path(args.out_dir)
    elif args.dataset_dir:
        out_path = Path(args.dataset_dir) / "raw"
    else:
        print("Error: Either --dataset-dir or --out-dir must be specified.", file=sys.stderr)
        return 2

    video_paths = []
    for v in args.videos:
        p = resolve_path(v)
        if not p.exists():
            print(f"Warning: Video not found: {p}")
            continue
        video_paths.append(p)

    if not video_paths:
        print("Error: No valid video files provided.")
        return 2

    try:
        cnt = extract_frames(
            video_paths=video_paths,
            out_dir=out_path,
            total_frames=int(args.num_frames),
            verbose=True,
        )
        print(f"Successfully extracted {cnt} frames to {out_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def handle_dataset_list(args) -> int:
    root = resolve_path(args.root) if args.root else resolve_path(appconfig.get_default_dataset_root())
    if not root.exists():
        print(f"Dataset root not found: {root}", file=sys.stderr)
        return 2
    datasets_found = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        md_path = entry / "METADATA.json"
        if not md_path.exists():
            continue
        try:
            md = json.loads(md_path.read_text(encoding="utf-8"))
        except Exception:
            md = {}
        model_cfg_path = entry / "models" / "model_config.json"
        try:
            mcfg = json.loads(model_cfg_path.read_text(encoding="utf-8")) if model_cfg_path.exists() else {}
        except Exception:
            mcfg = {}
        datasets_found.append({
            "uuid": md.get("uuid", entry.name),
            "name": md.get("name", ""),
            "created_at": (md.get("created_at", "")[:10] if md.get("created_at") else ""),
            "classes": len(md.get("class_names", [])),
            "task": mcfg.get("task", "-"),
            "trained": "yes" if model_cfg_path.exists() else "no",
        })
    if not datasets_found:
        print(f"No datasets found under: {root}")
        return 0
    print(f"Datasets in: {root}\n")
    col = "{:<38} {:<24} {:<12} {:>7} {:<8} {:<7}"
    print(col.format("UUID", "Name", "Created", "Classes", "Task", "Trained"))
    print("-" * 102)
    for d in datasets_found:
        print(col.format(d["uuid"], d["name"][:23], d["created_at"], d["classes"], d["task"], d["trained"]))
    print(f"\n{len(datasets_found)} dataset(s) found.")
    return 0


def handle_dataset_info(args) -> int:
    dataset_dir = resolve_path(args.dataset_dir)
    md_path = dataset_dir / "METADATA.json"
    if not md_path.exists():
        print(f"Error: Not a valid dataset directory (METADATA.json missing): {dataset_dir}", file=sys.stderr)
        return 2
    try:
        md = json.loads(md_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error: Could not read METADATA.json: {e}", file=sys.stderr)
        return 2
    print(f"Dataset : {dataset_dir}")
    print(f"UUID    : {md.get('uuid', '-')}")
    print(f"Name    : {md.get('name', '-')}")
    print(f"Created : {md.get('created_at', '-')}")
    classes = md.get("class_names", [])
    print(f"Classes : {len(classes)}  {classes}")
    if md.get("notes"):
        print(f"Notes   : {md['notes']}")
    print()
    # Split stats
    coco_dir = dataset_dir / "coco"
    for split in ("train", "valid", "test"):
        ann_path = coco_dir / split / "_annotations.coco.json"
        if ann_path.exists():
            try:
                data = json.loads(ann_path.read_text(encoding="utf-8"))
                n_img = len(data.get("images", []))
                n_ann = len(data.get("annotations", []))
                print(f"  {split:<6}: {n_img:>5} images, {n_ann:>6} annotations")
            except Exception:
                print(f"  {split:<6}: (could not read)")
        else:
            print(f"  {split:<6}: (no annotations)")
    # Model status
    model_cfg_path = dataset_dir / "models" / "model_config.json"
    print()
    if model_cfg_path.exists():
        try:
            mcfg = json.loads(model_cfg_path.read_text(encoding="utf-8"))
            print(f"Trained : yes  (task={mcfg.get('task','-')}, size={mcfg.get('size','-')}, classes={mcfg.get('num_classes','-')})")
        except Exception:
            print("Trained : yes  (model_config.json unreadable)")
    else:
        print("Trained : no")
    # Exports
    exports_dir = dataset_dir / "exports"
    if exports_dir.exists():
        exported = [f.name for f in sorted(exports_dir.iterdir()) if f.is_file()]
        if exported:
            print(f"Exports : {', '.join(exported)}")
    return 0


def handle_train(args) -> int:
    try:
        aug_cfg = None
        if getattr(args, "aug_config", None):
            aug_cfg = _load_jsonish(str(args.aug_config))
        # Resolve num_workers: explicit arg > appconfig default
        num_workers = args.num_workers if args.num_workers is not None else appconfig.get_default_num_workers()
        cfg = TrainConfig(
            dataset_dir=Path(args.dataset_dir),
            task=args.task,
            size=args.size,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            grad_accum=int(args.grad_accum),
            lr=float(args.lr),
            device=args.device,
            num_workers=num_workers,
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
    except KeyboardInterrupt:
        print("\nTraining interrupted.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def handle_export(args) -> int:
    dataset_dir = Path(args.dataset_dir)
    weights = Path(args.weights)
    out = Path(args.output) if args.output else None
    trained_model_cfg = _load_trained_model_config(dataset_dir)
    task = str(args.task or trained_model_cfg.get("task") or "detect").strip().lower()
    size = str(args.size or trained_model_cfg.get("size") or "nano").strip().lower()
    # Resolve format: explicit arg > appconfig default
    fmt = args.format or appconfig.get_default_export_format()
    # Validate --quantize usage
    if args.quantize and fmt == "tensorrt":
        print("Error: --quantize is not supported with --format tensorrt. "
              "Use --format onnx_quantized to export a quantized ONNX model.", file=sys.stderr)
        return 2

    if fmt in {"onnx", "onnx_quantized", "onnx_fp16"}:
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
            half=(fmt == "onnx_fp16" or args.fp16),
        )
        if not res.ok:
            print(f"Error: {res.message}", file=sys.stderr)
            return 2

        if fmt == "onnx_quantized" or args.quantize:
            q_res = quantize_onnx(
                onnx_path=res.output_path,
                output_path=None,
                dataset_dir=dataset_dir,
                calibration_split=args.calibration_split,
                calibration_count=args.calibration_count,
                height=int(args.height),
                width=int(args.width),
            )
            if not q_res.ok:
                print(f"Error: {q_res.message}", file=sys.stderr)
                return 2
            print(q_res.message)
            return 0

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


def handle_bundle(args) -> int:
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
        quantize=bool(args.quantize),
        calibration_split=args.calibration_split,
        calibration_count=int(args.calibration_count),
    )
    if not res.ok:
        print(f"Error: {res.message}", file=sys.stderr)
        return 2
    print(res.message)
    return 0


def handle_infer(args) -> int:
    bundle_dir = Path(args.bundle_dir)
    image_path = Path(args.image)
    weights = Path(args.weights) if args.weights else None
    # Resolve backend: explicit arg > appconfig default
    backend = args.backend or appconfig.get_default_inference_backend()
    try:
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
            backend=str(backend),
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
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

                out_img = resolve_path(args.out_image)
                out_img.parent.mkdir(parents=True, exist_ok=True)
                img.save(str(out_img))
                print(f"Wrote: {out_img}")
            except Exception as e:
                print(f"Warning: failed to write --out-image ({e})", file=sys.stderr)

    if args.out_json:
        outp = resolve_path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(res.payload, indent=2), encoding="utf-8")
        print(f"Wrote: {outp}")
    else:
        print(json.dumps(res.payload, indent=2))
    return 0


def handle_config(args) -> int:
    if args.config_cmd == "show":
        cfg_data = appconfig.load_config()
        print(f"Config file  : {appconfig.config_path()}")
        print(f"Config exists: {appconfig.config_path().exists()}")
        print()
        print("Effective settings:")

        def _show(key: str, value: object, env_var: str, fallback_note: str = "") -> None:
            env_val = os.environ.get(env_var)
            source = f"[env: {env_var}]" if env_val else ("(config file)" if key in cfg_data else f"(fallback{': ' + fallback_note if fallback_note else ''})")
            print(f"  {key:<28} = {value}  {source}")

        _show("default_dataset_root", appconfig.get_default_dataset_root(), appconfig.ENV_DATASETS, "datasets/")
        _show("default_num_workers",  appconfig.get_default_num_workers(),  appconfig.ENV_NUM_WORKERS, "0 on Windows, 4 elsewhere")
        _show("inference_backend",    appconfig.get_default_inference_backend(), appconfig.ENV_BACKEND, "auto")
        _show("export_format",        appconfig.get_default_export_format(), appconfig.ENV_EXPORT_FORMAT, "onnx")
        return 0

    if args.config_cmd == "set":
        if args.config_set_cmd == "dataset-root":
            resolved = str(Path(args.path).expanduser())
            appconfig.set_default_dataset_root(resolved)
            print(f"default_dataset_root → {resolved}")
        elif args.config_set_cmd == "num-workers":
            appconfig.set_default_num_workers(int(args.value))
            print(f"default_num_workers → {args.value}")
        elif args.config_set_cmd == "inference-backend":
            try:
                appconfig.set_default_inference_backend(args.value)
                print(f"inference_backend → {args.value}")
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 2
        elif args.config_set_cmd == "export-format":
            try:
                appconfig.set_default_export_format(args.value)
                print(f"export_format → {args.value}")
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 2
        print(f"Saved to: {appconfig.config_path()}")
        return 0

    return 2
