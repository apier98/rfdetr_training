"""Business-logic handlers for each CLI command.

Each function takes the parsed ``argparse.Namespace`` and returns an integer
exit code (0 = success, 1 = interrupted, 2 = error, 3 = validation warning).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
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
from .videos import compute_frames_for_fps, extract_frames, scan_video_dir
from .pathutil import resolve_path


def _infer_predictive_scope(rows: List[dict]) -> dict:
    """Infer a single training scope from dataset rows when it is unambiguous."""
    mold_ids = sorted({r.get("mold_id") for r in rows if r.get("mold_id")})
    material_ids = sorted({r.get("material_id") for r in rows if r.get("material_id")})
    machine_ids = sorted({r.get("machine_id") for r in rows if r.get("machine_id")})
    machine_families = sorted(
        {
            (r.get("context") or {}).get("machine_family")
            for r in rows
            if (r.get("context") or {}).get("machine_family")
        }
    )
    return {
        "mold_id": mold_ids[0] if len(mold_ids) == 1 else None,
        "material_id": material_ids[0] if len(material_ids) == 1 else None,
        "machine_id": (
            machine_ids[0]
            if len(machine_ids) == 1
            else (machine_families[0] if len(machine_families) == 1 else None)
        ),
    }


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
    print(
        "Note: Label Studio exports YOLO files with class IDs sorted ALPHABETICALLY by label name,\n"
        "  regardless of the order shown in the Label Studio UI.\n"
        "  Example: if your classes are Component_Base, Weld_Line, Sink_Mark, Label Studio assigns\n"
        "    class 0 = Component_Base, class 1 = Sink_Mark, class 2 = Weld_Line  (alphabetical)\n"
        "  while METADATA.json might have them in a different (non-alphabetical) order.\n"
        "  Fix: in your Label Studio labeling config, add category=\"N\" to pin each label to its ID:\n"
        "    <Label value=\"Component_Base\" category=\"0\"/>\n"
        "    <Label value=\"Weld_Line\"      category=\"1\"/>\n"
        "    <Label value=\"Sink_Mark\"      category=\"2\"/>\n"
        "  Verify: compare METADATA.json class_names order with the order in a YOLO .txt file\n"
        "  before training to avoid silent class swaps."
    )
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


def _print_split_class_stats(split_dir: Path, label: str) -> None:
    """Print per-class instance/image counts for a COCO split."""
    from collections import Counter, defaultdict
    from .jsonutil import load_json

    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return
    try:
        coco = load_json(ann_path)
    except Exception:
        return

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}

    ann_img_ids: set = {int(a["image_id"]) for a in anns}
    bg_count = sum(1 for im in images if int(im.get("id", -1)) not in ann_img_ids)

    cat_insts: Counter = Counter(int(a["category_id"]) for a in anns)
    cat_imgs: defaultdict = defaultdict(set)
    for a in anns:
        cat_imgs[int(a["category_id"])].add(int(a["image_id"]))

    print(f"  [{label}] {len(images)} images")
    max_name_len = max((len(n) for n in cats.values()), default=12)
    for cid, name in sorted(cats.items(), key=lambda x: x[1]):
        insts = cat_insts.get(cid, 0)
        imgs = len(cat_imgs.get(cid, set()))
        print(f"    {name:<{max_name_len}}  {insts:>5} inst  {imgs:>4} imgs")
    if bg_count > 0:
        print(f"    {'(background)':<{max_name_len}}             {bg_count:>4} imgs")


def handle_dataset_subsample(args) -> int:
    dataset_dir = resolve_path(args.dataset_dir)
    coco_dir = dataset_dir / "coco"

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]
    any_error = False

    for split in splits:
        split_dir = coco_dir / split
        ann_path = split_dir / "_annotations.coco.json"

        if not ann_path.exists():
            if args.split == "all":
                continue  # silently skip missing splits when iterating all
            print(f"Error: {ann_path} not found", file=sys.stderr)
            return 2

        print(f"\n── {split} " + "─" * max(0, 40 - len(split)))
        _print_split_class_stats(split_dir, "before")

        res = subsample_coco_split(
            split_dir,
            fraction=args.fraction,
            max_images=args.max_images,
            min_instances_per_class=args.min_instances,
            seed=args.seed,
            dry_run=bool(args.dry_run),
        )

        if not res.ok:
            print(f"  Error: {res.message}", file=sys.stderr)
            any_error = True
            continue

        print(f"  {res.message}")
        if res.backup_path is not None:
            print(f"  Backup: {res.backup_path}")

        if not args.dry_run:
            _print_split_class_stats(split_dir, "after")

    return 2 if any_error else 0


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

    video_paths: List[Path] = []

    for v in (args.videos or []):
        p = resolve_path(v)
        if not p.exists():
            print(f"Warning: Video not found: {p}", file=sys.stderr)
        else:
            video_paths.append(p)

    if getattr(args, "videos_dir", None):
        vd = resolve_path(args.videos_dir)
        if not vd.is_dir():
            print(f"Error: --videos-dir is not a directory: {vd}", file=sys.stderr)
            return 2
        raw_exts = getattr(args, "ext", None) or "mp4,avi,mov,mkv,webm"
        exts = {"." + e.strip().lstrip(".").lower() for e in raw_exts.split(",") if e.strip()}
        found = scan_video_dir(vd, exts)
        if not found:
            print(f"Warning: No video files found in {vd} with extensions: {', '.join(sorted(exts))}", file=sys.stderr)
        video_paths.extend(found)

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[Path] = []
    for p in video_paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    video_paths = unique

    if not video_paths:
        print("Error: No valid video files provided. Use --videos or --videos-dir.", file=sys.stderr)
        return 2

    fps_val = getattr(args, "fps", None)
    num_frames_val = getattr(args, "num_frames", None)
    if fps_val is not None and fps_val > 0:
        total_frames = compute_frames_for_fps(video_paths, fps_val)
        print(f"Target: {fps_val} fps → ~{total_frames} frames across {len(video_paths)} video(s)")
    else:
        total_frames = int(num_frames_val) if num_frames_val is not None else 100
        print(f"Target: {total_frames} frames across {len(video_paths)} video(s)")

    try:
        cnt = extract_frames(
            video_paths=video_paths,
            out_dir=out_path,
            total_frames=total_frames,
            verbose=True,
        )
        print(f"\nExtracted {cnt} frames → {out_path}")
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


def _resolve_finetune_weights(bundle_id: str, args) -> str:
    """Resolve a bundle_id to a checkpoint path from the lake model registry."""
    task = getattr(args, "task", "detect")
    task_dir = "defect_detection" if task == "detect" else "monitor_segmentation"

    # Try to find the lake root from config files
    lake_root = None
    for candidate in [
        Path("data_lake_config.json"),
        Path.home() / ".aria" / "data_lake_config.json",
    ]:
        if candidate.exists():
            cfg = json.loads(candidate.read_text(encoding="utf-8"))
            lake_root = Path(cfg.get("root", ""))
            break

    if lake_root is None:
        lake_root = Path(getattr(args, "lake_root", "."))

    registry_path = lake_root / "models" / task_dir / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Lake model registry not found at {registry_path}. "
            f"Cannot resolve bundle_id '{bundle_id}'."
        )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    bundles = registry.get("bundles", [])

    # Handle special "latest" keyword
    if bundle_id == "latest":
        active = registry.get("active", {})
        bundle_id = active.get("stable") or active.get("dev")
        if not bundle_id:
            raise ValueError("No active bundle found in registry for 'latest'.")

    # Find the bundle entry
    match = None
    for b in bundles:
        if b.get("bundle_id") == bundle_id:
            match = b
            break

    if match is None:
        raise ValueError(
            f"Bundle '{bundle_id}' not found in registry at {registry_path}. "
            f"Available: {[b.get('bundle_id') for b in bundles]}"
        )

    # Resolve checkpoint path
    bundle_dir = Path(match.get("path", ""))
    if not bundle_dir.is_absolute():
        bundle_dir = lake_root / bundle_dir

    for ckpt_name in ("checkpoint_portable.pth", "checkpoint.pth", "checkpoint_best_total.pth"):
        ckpt = bundle_dir / ckpt_name
        if ckpt.exists():
            print(f"Resolved finetune-from '{match.get('bundle_id')}' → {ckpt}")
            return str(ckpt)

    raise FileNotFoundError(
        f"No checkpoint found in bundle directory {bundle_dir}. "
        f"Looked for: checkpoint_portable.pth, checkpoint.pth, checkpoint_best_total.pth"
    )


def handle_train(args) -> int:
    try:
        aug_cfg = None
        if getattr(args, "aug_config", None):
            aug_cfg = _load_jsonish(str(args.aug_config))
        # Resolve finetune-from: if not an existing path, treat as bundle_id
        finetune_weights = getattr(args, "finetune_from", None)
        if finetune_weights and not Path(finetune_weights).exists():
            finetune_weights = _resolve_finetune_weights(finetune_weights, args)
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
            finetune_from=finetune_weights,
            use_checkpoint_model=bool(args.use_checkpoint_model),
            checkpoint_key=args.checkpoint_key,
            patch_inference_mode=args.patch_inference_mode,
            validate_dataset=(not bool(args.no_validate)),
            multi_scale=getattr(args, "multi_scale", None),
            expanded_scales=getattr(args, "expanded_scales", None),
            do_random_resize_via_padding=getattr(args, "do_random_resize_via_padding", None),
            aug_config=aug_cfg,
            no_aug=bool(getattr(args, "no_aug", False)),
            log_file=getattr(args, "log_file", None),
            no_log_file=bool(getattr(args, "no_log_file", False)),
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
        make_mpk=bool(args.mpk),
        overwrite=bool(args.overwrite),
        quantize=bool(args.quantize),
        calibration_split=args.calibration_split,
        calibration_count=int(args.calibration_count),
        bundle_id=args.bundle_id,
        model_name=args.model_name,
        model_version=args.model_version,
        channel=args.channel,
        supersedes=args.supersedes,
        min_app_version=args.min_app_version,
        standalone=bool(args.standalone),
    )
    if not res.ok:
        print(f"Error: {res.message}", file=sys.stderr)
        return 2
    print(res.message)

    # E3: Publish-After-Train — if --publish, immediately publish the bundle.
    if getattr(args, "publish", False):
        role = getattr(args, "publish_role", None)
        if not role:
            print("Error: --publish-role is required when --publish is set", file=sys.stderr)
            return 2
        from moldvision.publish import publish_bundle
        try:
            pub_result = publish_bundle(
                res.bundle_dir,
                role=role,
                channel=args.channel,
                dry_run=getattr(args, "publish_dry_run", False),
            )
        except Exception as e:
            print(f"Error: Publishing failed: {e}", file=sys.stderr)
            return 2
        import json as _json
        if getattr(args, "publish_dry_run", False):
            print("DRY RUN — catalog entry:")
        else:
            print("Published successfully:")
        print(_json.dumps(pub_result, indent=2))
    return 0


def _draw_infer_overlay(img_path: Path, res, mask_alpha: float, out_path: Path) -> None:
    """Draw detection/segmentation overlay and save to out_path. Best-effort."""
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:
        return
    try:
        img = Image.open(str(img_path)).convert("RGB")
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
                    mask_img = Image.fromarray((mm.astype(np.uint8) * int(round(mask_alpha * 255))))
                    overlay.paste((int(r), int(g), int(b), int(round(mask_alpha * 255))), (0, 0), mask_img)
                img = Image.alpha_composite(rgba, overlay).convert("RGB")
            except Exception:
                pass
        draw = ImageDraw.Draw(img)
        if res.boxes and res.scores and res.labels:
            for b, sc, lab in zip(res.boxes, res.scores, res.labels):
                x1, y1, x2, y2 = [float(x) for x in b]
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                draw.text((x1 + 2, max(0, y1 - 12)), f"{int(lab)}:{float(sc):.2f}", fill=(0, 255, 0))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
    except Exception:
        pass


def _handle_batch_infer(args) -> int:
    """Run inference on all images in a directory."""
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    bundle_dir = Path(args.bundle_dir)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: --input-dir is not a directory: {input_dir}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else (input_dir / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 2

    weights = Path(args.weights) if args.weights else None
    backend = args.backend or appconfig.get_default_inference_backend()
    mask_alpha = float(args.mask_alpha) if args.mask_alpha is not None else 0.45
    draw_overlays = bool(getattr(args, "overlays", False))

    summary = []
    errors = 0
    for img_path in images:
        try:
            res = infer_from_bundle(
                bundle_dir=bundle_dir,
                image_path=img_path,
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
            print(f"Error [{img_path.name}]: {e}", file=sys.stderr)
            errors += 1
            continue

        if not res.ok or res.payload is None:
            print(f"Error [{img_path.name}]: {res.message}", file=sys.stderr)
            errors += 1
            continue

        json_path = out_dir / f"{img_path.stem}.json"
        json_path.write_text(json.dumps(res.payload, indent=2), encoding="utf-8")

        if draw_overlays:
            overlay_path = out_dir / f"{img_path.stem}_overlay{img_path.suffix}"
            _draw_infer_overlay(img_path, res, mask_alpha, overlay_path)

        n_det = len(res.boxes or [])
        print(f"[{img_path.name}] {n_det} detection(s) → {json_path.name}")
        summary.append({"image": img_path.name, "detections": n_det})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    total = len(images)
    ok = total - errors
    print(f"\nBatch complete: {ok}/{total} succeeded. Results in: {out_dir}")
    return 0 if errors == 0 else 3


def handle_infer(args) -> int:
    if getattr(args, "input_dir", None):
        return _handle_batch_infer(args)

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
        alpha = float(args.mask_alpha) if args.mask_alpha is not None else 0.45
        out_img = resolve_path(args.out_image)
        _draw_infer_overlay(image_path, res, alpha, out_img)
        if out_img.exists():
            print(f"Wrote: {out_img}")
        else:
            print(f"Warning: failed to write --out-image", file=sys.stderr)

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
        _show("predictive_runs_root", appconfig.get_predictive_runs_root(), appconfig.ENV_PREDICTIVE_RUNS, str(appconfig.config_dir() / "predictive_runs"))
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
        elif args.config_set_cmd == "predictive-runs-root":
            resolved = str(Path(args.path).expanduser())
            appconfig.set_predictive_runs_root(resolved)
            print(f"predictive_runs_root → {resolved}")
        print(f"Saved to: {appconfig.config_path()}")
        return 0

    return 2


# ──────────────────────────────────────────────────────────────────────────────
# lake command dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def _lake_cfg(args) -> "LakeConfig":  # type: ignore[name-defined]
    from .lake import LakeConfig
    override = Path(args.lake_root) if getattr(args, "lake_root", None) else None
    return LakeConfig.find(override)


def handle_lake(args) -> int:
    lake_cmd = getattr(args, "lake_cmd", None)

    if lake_cmd == "init":
        return _handle_lake_init(args)

    if lake_cmd == "import":
        return _handle_lake_external_import(args)

    if lake_cmd == "session":
        sub = getattr(args, "lake_session_cmd", None)
        if sub == "import":
            return _handle_lake_session_import(args)
        if sub == "list":
            return _handle_lake_session_list(args)
        if sub == "mark-bg":
            return _handle_lake_session_mark_bg(args)

    if lake_cmd == "label-batch":
        sub = getattr(args, "lake_label_batch_cmd", None)
        if sub == "create":
            return _handle_lake_label_batch_create(args)
        if sub == "commit":
            return _handle_lake_label_batch_commit(args)
        if sub == "status":
            return _handle_lake_label_batch_status(args)

    if lake_cmd == "pull":
        return _handle_lake_pull(args)

    if lake_cmd == "index":
        return _handle_lake_index(args)

    if lake_cmd == "models":
        sub = getattr(args, "lake_models_cmd", None)
        if sub == "install":
            return _handle_lake_models_install(args)
        if sub == "list":
            return _handle_lake_models_list(args)
        if sub == "promote":
            return _handle_lake_models_promote(args)

    if lake_cmd == "pools":
        sub = getattr(args, "lake_pools_cmd", None)
        if sub == "add-hard-negative":
            return _handle_lake_pools_add(args, pool="hard_negatives")
        if sub == "add-background":
            return _handle_lake_pools_add(args, pool="backgrounds")

    from . import cli as _cli
    _cli.build_parser().parse_args(["lake", "--help"])
    return 2


def _handle_lake_init(args) -> int:
    from .lake import LakeConfig, init_lake
    root = Path(args.root) if getattr(args, "root", None) else LakeConfig.default_root()
    already_exists = (root / "data_lake_config.json").exists()
    cfg = init_lake(root)
    if already_exists:
        print(f"Lake already initialised — refreshed skeleton at: {cfg.root}")
    else:
        print(f"Lake initialised at: {cfg.root}")
    print(f"  Config : {cfg.root / 'data_lake_config.json'}")
    print(f"  Index  : {cfg.root / 'image_index.jsonl'}")
    print(f"  Tip    : set ARIA_DATA_LAKE={cfg.root} (or use --lake-root on every command)")
    return 0


def _handle_lake_external_import(args) -> int:
    from .lake import external_import
    cfg = _lake_cfg(args)
    coco_path = Path(args.coco_json) if args.coco_json else None
    try:
        result = external_import(
            cfg,
            images_dir=Path(args.images_dir),
            task=args.task,
            coco_json=coco_path,
            session_id=getattr(args, "session_id", None),
            name=getattr(args, "name", None),
            machine_id=getattr(args, "machine_id", None),
            mold_id=getattr(args, "mold_id", None),
            part_id=getattr(args, "part_id", None),
            notes=getattr(args, "notes", None),
            overwrite=bool(args.overwrite),
        )
        verb = "Updated" if result.already_existed else "Imported"
        print(f"{verb} external session: {result.session_id}")
        print(f"  Images added:    {result.images_added}")
        print(f"  Labeled:         {result.images_labeled}")
        print(f"  Unlabeled:       {result.images_unlabeled}")
        if result.images_unlabeled and not args.coco_json:
            print(f"  Tip: use 'lake label-batch create --sessions {result.session_id}' to annotate the unlabeled images.")
        return 0
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    root_arg = getattr(args, "root", None)
    from .lake import LakeConfig
    root = Path(root_arg) if root_arg else LakeConfig.default_root()
    try:
        cfg = init_lake(root)
        print(f"Data lake initialised at: {cfg.root}")
        print(f"Config: {cfg.root / 'data_lake_config.json'}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_session_import(args) -> int:
    from .lake import LakeConfig
    from .lake import session_import
    cfg = _lake_cfg(args)
    try:
        result = session_import(
            cfg,
            session_meta_path=Path(args.session_meta),
            inspection_frames_dir=Path(args.inspection_frames) if args.inspection_frames else None,
            monitor_frames_dir=Path(args.monitor_frames) if args.monitor_frames else None,
            overwrite=bool(args.overwrite),
        )
        verb = "Updated" if result.already_existed else "Imported"
        print(f"{verb} session: {result.session_id}")
        print(f"  Inspection frames: {result.inspection_frames_added}")
        print(f"  Monitor frames:    {result.monitor_frames_added}")
        return 0
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_session_list(args) -> int:
    from .lake import session_list
    cfg = _lake_cfg(args)
    try:
        label_status = getattr(args, "label_status", None)
        if label_status == "any":
            label_status = None
        session_list(
            cfg,
            machine_id=getattr(args, "machine_id", None),
            mold_id=getattr(args, "mold_id", None),
            part_id=getattr(args, "part_id", None),
            from_date=getattr(args, "from_date", None),
            to_date=getattr(args, "to_date", None),
            task=getattr(args, "task", None),
            label_status=label_status,
            marker=getattr(args, "marker", None),
            min_frames=getattr(args, "min_frames", 0) or 0,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_session_mark_bg(args) -> int:
    from .lake_label import session_mark_backgrounds
    cfg = _lake_cfg(args)
    try:
        session_mark_backgrounds(
            cfg,
            session_id=args.session,
            task=args.task,
            dry_run=bool(getattr(args, "dry_run", False)),
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_label_batch_create(args) -> int:
    from .lake_label import label_batch_create
    cfg = _lake_cfg(args)
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()] if args.sessions else None
    try:
        label_batch_create(
            cfg,
            task=args.task,
            sessions=sessions,
            all_sessions=bool(args.all_sessions),
            machine_id=getattr(args, "machine_id", None),
            mold_id=getattr(args, "mold_id", None),
            marker=getattr(args, "marker", None),
            only_unlabeled=bool(args.only_unlabeled),
            n=int(args.n),
            sample_mode=args.sample_mode,
            min_frame_gap=int(args.min_frame_gap),
            skip_first=int(args.skip_first),
            skip_last=int(args.skip_last),
            seed=int(args.seed),
            batch_name=getattr(args, "batch_name", None),
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_label_batch_commit(args) -> int:
    from .lake_label import label_batch_commit
    cfg = _lake_cfg(args)
    coco_path = Path(args.coco_json) if args.coco_json else None
    try:
        label_batch_commit(
            cfg,
            batch_id=args.batch_id,
            coco_json_path=coco_path,
            dry_run=bool(args.dry_run),
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_label_batch_status(args) -> int:
    from .lake_label import label_batch_status
    cfg = _lake_cfg(args)
    try:
        label_batch_status(cfg, task=getattr(args, "task", None))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_pull(args) -> int:
    from .lake_pull import lake_pull
    cfg = _lake_cfg(args)
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()] if args.sessions else None
    priority_sessions = (
        [s.strip() for s in args.priority_sessions.split(",") if s.strip()]
        if getattr(args, "priority_sessions", None)
        else None
    )
    ds_root = Path(args.dataset_root) if args.dataset_root else None
    try:
        lake_pull(
            cfg,
            task=args.task,
            sessions=sessions,
            all_sessions=bool(args.all_sessions),
            machine_id=getattr(args, "machine_id", None),
            mold_id=getattr(args, "mold_id", None),
            part_id=getattr(args, "part_id", None),
            from_date=getattr(args, "from_date", None),
            to_date=getattr(args, "to_date", None),
            marker=getattr(args, "marker", None),
            include_hard_negatives=bool(args.include_hard_negatives),
            include_backgrounds=bool(args.include_backgrounds),
            total=getattr(args, "total", None),
            max_per_session=args.max_per_session,
            min_per_session=int(args.min_per_session),
            priority_sessions=priority_sessions,
            priority_weight=float(getattr(args, "priority_weight", 3.0)),
            balance_classes=bool(args.balance_classes),
            min_per_class=args.min_per_class,
            train_ratio=float(args.train_ratio),
            seed=int(args.seed),
            dataset_uuid=getattr(args, "dataset_uuid", None),
            dataset_name=getattr(args, "dataset_name", None),
            dataset_root=ds_root,
            dry_run=bool(args.dry_run),
        )
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_index(args) -> int:
    from .lake import index_rebuild, index_stats
    cfg = _lake_cfg(args)
    try:
        if args.rebuild:
            n = index_rebuild(cfg)
            print(f"Index rebuilt: {n} records written to image_index.jsonl")
        elif args.stats:
            index_stats(cfg, task=getattr(args, "task", None))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_models_install(args) -> int:
    from .lake_models import models_install
    cfg = _lake_cfg(args)
    try:
        models_install(cfg, bundle_path=Path(args.bundle), task=args.task)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_models_list(args) -> int:
    from .lake_models import models_list
    cfg = _lake_cfg(args)
    try:
        models_list(cfg, task=getattr(args, "task", None))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_models_promote(args) -> int:
    from .lake_models import models_promote
    cfg = _lake_cfg(args)
    try:
        models_promote(cfg, bundle_id=args.bundle_id, task=args.task, channel=args.channel)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_lake_pools_add(args, *, pool: str) -> int:
    """Add images to ``hard_negatives`` or ``backgrounds`` pool."""
    from .lake import (
        LABEL_STATUS_BACKGROUND,
        LABEL_STATUS_HARD_NEGATIVE,
        load_index,
        patch_index_records,
    )
    cfg = _lake_cfg(args)
    images: List[str] = list(args.images)
    if not images:
        print("Error: provide at least one --image path", file=sys.stderr)
        return 2

    reason = getattr(args, "reason", "")
    manifest_path = cfg.storage().abs_path(f"pools/{pool}/manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    new_status = LABEL_STATUS_HARD_NEGATIVE if pool == "hard_negatives" else LABEL_STATUS_BACKGROUND
    added = 0
    # Load existing manifest to check for duplicates
    existing: set = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    existing.add(json.loads(line).get("rel_path", ""))
                except Exception:
                    pass

    with manifest_path.open("a", encoding="utf-8") as fh:
        for img_rel in images:
            # Normalise to POSIX relative path
            img_rel = img_rel.replace("\\", "/")
            if img_rel in existing:
                print(f"  Skipped (already in pool): {img_rel}")
                continue
            entry: dict = {"rel_path": img_rel}
            if reason:
                entry["reason"] = reason
            fh.write(json.dumps(entry) + "\n")
            added += 1

    # Update image_index.jsonl for known images
    all_recs = load_index(cfg.root)
    known_paths = {r["rel_path"] for r in all_recs}
    to_patch = [img.replace("\\", "/") for img in images if img.replace("\\", "/") in known_paths]
    forced_task = getattr(args, "task", None)
    if to_patch:
        for img_norm in to_patch:
            if forced_task:
                status_field = "detect_status" if forced_task == "detect" else "seg_status"
            else:
                # Infer from path: monitor_frames → seg, inspection_frames or unknown → detect
                status_field = "seg_status" if "monitor_frames" in img_norm else "detect_status"
            patch_index_records(cfg.root, [img_norm], {status_field: new_status})

    print(f"Added {added} image(s) to {pool} pool.")
    return 0


# ── predictive ──────────────────────────────────────────────────────────────

def handle_predictive(args) -> int:
    subcmd = getattr(args, "predictive_cmd", None)
    if subcmd == "list-artifacts":
        return _handle_predictive_list_artifacts(args)
    if subcmd == "validate-dataset":
        return _handle_predictive_validate_dataset(args)
    if subcmd == "train":
        return _handle_predictive_train(args)
    if subcmd == "bundle":
        return _handle_predictive_bundle(args)
    print(f"Unknown predictive sub-command: {subcmd}")
    return 2


def _handle_predictive_list_artifacts(args) -> int:
    shared_root = (
        resolve_path(args.shared_root)
        if getattr(args, "shared_root", None)
        else appconfig.get_shared_root()
    )
    if shared_root is None:
        print("ERROR: ARIA shared root is not configured.")
        print("       Set ARIA_SHARED_ROOT or pass --shared-root.")
        return 2

    exports_root = shared_root / "ingest" / "moldtrace" / "training_rows" / "v1"
    records = _collect_predictive_artifacts(exports_root)
    if args.session_id:
        records = [r for r in records if r["session_id"] == args.session_id]
    if args.mold_id:
        records = [r for r in records if r["mold_id"] == args.mold_id]
    if args.material_id:
        records = [r for r in records if r["material_id"] == args.material_id]
    if args.machine_family:
        records = [r for r in records if r["machine_family"] == args.machine_family]

    limit = max(1, int(getattr(args, "limit", 50)))
    records = records[:limit]
    if getattr(args, "json", False):
        print(json.dumps(records, indent=2))
        return 0

    print(f"Found {len(records)} predictive artifact(s) under {exports_root}")
    for record in records:
        print(
            f"- {record['export_id']}  staged={record['staged_at_utc']}  "
            f"mold={record['mold_id'] or '-'}  material={record['material_id'] or '-'}  "
            f"family={record['machine_family'] or '-'}  "
            f"rows={record['rows_total']}  ready={record['rows_training_ready']}  "
            f"gate={record['gate_status']}"
        )
        print(f"  input={record['training_rows_path']}")
        print(f"  session={record['session_id']}  root={record['export_root']}")
    return 0


def _collect_predictive_artifacts(exports_root: Path) -> list[dict]:
    if not exports_root.exists():
        return []

    records: list[dict] = []
    for session_dir in sorted((p for p in exports_root.iterdir() if p.is_dir()), reverse=True):
        for export_dir in sorted((p for p in session_dir.iterdir() if p.is_dir()), reverse=True):
            manifest = _read_optional_json(export_dir / "export_manifest.json")
            if not manifest:
                continue
            session_meta = _read_optional_json(export_dir / "session.json")
            rows_summary = _read_optional_json(export_dir / "training_rows_summary.json")
            quality_gate = _read_optional_json(export_dir / "quality_gate_summary.json")
            counts = rows_summary.get("counts") if isinstance(rows_summary.get("counts"), dict) else {}
            gate_counts = quality_gate.get("counts") if isinstance(quality_gate.get("counts"), dict) else {}
            machine_context = {}
            critical_slot_policy = rows_summary.get("critical_slot_policy")
            if isinstance(critical_slot_policy, dict):
                machine_context = critical_slot_policy.get("machine_context") or {}
            hmi_layouts = rows_summary.get("features", {}).get("hmi_layouts_seen", [])
            first_layout = hmi_layouts[0] if hmi_layouts else {}
            records.append(
                {
                    "session_id": manifest.get("session_id") or session_dir.name,
                    "export_id": manifest.get("export_id") or export_dir.name,
                    "staged_at_utc": manifest.get("staged_at_utc") or "",
                    "mold_id": session_meta.get("mold_id"),
                    "material_id": session_meta.get("material_id"),
                    "machine_id": session_meta.get("machine_id"),
                    "machine_family": (
                        machine_context.get("machine_family")
                        or first_layout.get("machine_family")
                    ),
                    "layout_id": (
                        machine_context.get("layout_id")
                        or first_layout.get("hmi_layout_id")
                    ),
                    "rows_total": counts.get("rows_total", 0),
                    "rows_training_ready": counts.get(
                        "rows_training_ready",
                        quality_gate.get("outputs", {}).get("training_ready_count", 0),
                    ),
                    "components_total": gate_counts.get("components_total", 0),
                    "components_training_ready": gate_counts.get("components_training_ready", 0),
                    "gate_status": (quality_gate.get("gate") or {}).get("status", "unknown"),
                    "gate_passed": (quality_gate.get("gate") or {}).get("passed"),
                    "training_rows_path": str(export_dir / "training_rows.jsonl"),
                    "export_root": str(export_dir),
                }
            )

    records.sort(key=lambda record: (record["staged_at_utc"], record["export_id"]), reverse=True)
    return records


def _read_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _handle_predictive_validate_dataset(args) -> int:
    from .predictive.training_row_loader import (
        assess_training_readiness,
        filter_eligible,
        load_training_rows,
        summarize_scope_distribution,
        summarize_dataset,
        validate_dataset,
    )

    input_path = resolve_path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 2

    print(f"Loading {input_path} ...")
    try:
        rows = load_training_rows(input_path)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    if not rows:
        print("WARNING: File contains zero rows.")
        return 3

    if args.eligible_only:
        rows = filter_eligible(rows)
        print(f"Filtered to {len(rows)} eligible (training_ready) rows.")
        if not rows:
            print("WARNING: No eligible rows after filtering.")
            return 3

    report = validate_dataset(rows)
    print(f"\nValidation: {report['valid_rows']}/{report['total_rows']} rows valid.")
    if report["row_errors"]:
        print(f"\n{report['invalid_rows']} row(s) with errors:")
        for entry in report["row_errors"][:20]:
            sid = entry.get("session_id", "?")
            cid = entry.get("component_id", "?")
            for err in entry["errors"]:
                print(f"  [{entry['index']}] session={sid} component={cid}: {err}")
        if len(report["row_errors"]) > 20:
            print(f"  ... and {len(report['row_errors']) - 20} more")

    sw = report.get("scope_warnings", {})
    if sw.get("null_mold_id", 0) > 0:
        print(f"SCOPE WARNING: {sw['null_mold_id']} row(s) missing mold_id — scope-specific training will not work correctly.")
    if sw.get("null_material_id", 0) > 0:
        print(f"SCOPE WARNING: {sw['null_material_id']} row(s) missing material_id — pass --material-id when training to ensure correct scoping.")

    if args.summary:
        summary = summarize_dataset(rows)
        print(f"\n--- Dataset Summary ---")
        print(f"Total rows:        {summary['total_rows']}")
        print(f"Eligible rows:     {summary['eligible_rows']}")
        print(f"Feature columns:   {summary['feature_columns']}")
        print(f"Schema homogeneous:{summary['schema_homogeneous']}")
        qs = summary.get("quality_score_stats") or {}
        if qs.get("count"):
            print(f"Quality score:     min={qs['min']}  max={qs['max']}  mean={qs['mean']}  n={qs['count']}")
        dc = summary.get("defect_counts") or {}
        if dc:
            print("Defect counts:")
            for label, cnt in dc.items():
                print(f"  {label}: {cnt}")
        layouts = summary.get("hmi_layouts_seen") or []
        if layouts:
            print(f"HMI layouts seen:  {len(layouts)}")
            for lay in layouts:
                print(f"  {lay.get('hmi_layout_id', '?')} / {lay.get('machine_family', '?')}")
        sc = summary.get("scope_coverage") or {}
        if sc:
            print(f"Scope coverage:    {sc.get('distinct_scopes', 0)} distinct scope(s)")
            if sc.get("null_mold_id", 0):
                print(f"  ⚠  {sc['null_mold_id']} row(s) missing mold_id")
            if sc.get("null_material_id", 0):
                print(f"  ⚠  {sc['null_material_id']} row(s) missing material_id")
            for scope in sc.get("scopes", [])[:10]:
                print(f"  scope: mold_id={scope[0]!r}  material_id={scope[1]!r}")
            for family in sc.get("machine_families", [])[:10]:
                print(f"  machine_family: {family}")
        readiness = summary.get("training_readiness") or assess_training_readiness(summary["eligible_rows"])
        print(
            f"Training readiness:{readiness['level']:>12}  "
            f"{readiness['message']}"
        )

    return 0 if report["valid"] else 3


def _handle_predictive_train(args) -> int:
    """Handle ``predictive train``.

    Trains one LightGBM model per suggestion target (quality score + 4 defect
    classifiers) and writes ``train_result.pkl`` + ``training_meta.json`` to
    ``--output-dir``.
    """
    import pickle

    from .predictive.training_row_loader import (
        assess_training_readiness,
        check_schema_homogeneity,
        summarize_scope_distribution,
        load_training_rows,
        validate_dataset,
    )
    from .predictive.trainer import GbtTrainingConfig, train_suggestion_models

    input_path = resolve_path(args.input)
    scope_mold_id: str | None = getattr(args, "mold_id", None)
    scope_material_id: str | None = getattr(args, "material_id", None)
    scope_machine_id: str | None = getattr(args, "machine_id", None)

    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 2

    print(f"Loading {input_path} ...")
    try:
        rows = load_training_rows(input_path)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if not rows:
        print("ERROR: File contains zero rows.")
        return 2

    report = validate_dataset(rows)
    if not report["valid"]:
        print(
            f"ERROR: Dataset contains {report['invalid_rows']} invalid row(s). "
            "Run 'moldvision predictive validate-dataset --summary' and fix the input before training."
        )
        for entry in report["row_errors"][:10]:
            sid = entry.get("session_id", "?")
            cid = entry.get("component_id", "?")
            for err in entry["errors"]:
                print(f"  [{entry['index']}] session={sid} component={cid}: {err}")
        if len(report["row_errors"]) > 10:
            print(f"  ... and {len(report['row_errors']) - 10} more")
        return 2

    scope_distribution = summarize_scope_distribution(rows)
    if scope_distribution["distinct_scope_count"] > 1 and not (scope_mold_id or scope_material_id):
        print(
            "WARNING: Dataset contains multiple mold/material scopes. "
            "For production training, prefer one JSONL per scope or pass --mold-id/--material-id explicitly."
        )

    inferred_scope = _infer_predictive_scope(rows)
    if scope_mold_id is None:
        scope_mold_id = inferred_scope["mold_id"]
    if scope_material_id is None:
        scope_material_id = inferred_scope["material_id"]
    if scope_machine_id is None:
        scope_machine_id = inferred_scope["machine_id"]

    output_dir = (
        resolve_path(args.output_dir)
        if getattr(args, "output_dir", None)
        else _default_predictive_train_output_dir(
            input_path,
            mold_id=scope_mold_id,
            material_id=scope_material_id,
            machine_id=scope_machine_id,
        )
    )
    if not getattr(args, "output_dir", None):
        print(f"Output directory not provided; using local predictive run folder: {output_dir}")

    # Scope preflight / filtering.
    if scope_mold_id or scope_material_id or scope_machine_id:
        def _matches_scope(row: dict) -> bool:
            if scope_mold_id and row.get("mold_id") != scope_mold_id:
                return False
            if scope_material_id and row.get("material_id") != scope_material_id:
                return False
            if scope_machine_id and (row.get("context") or {}).get("machine_family") != scope_machine_id:
                return False
            return True

        matching_rows = [r for r in rows if _matches_scope(r)]
        dropped = len(rows) - len(matching_rows)
        if dropped and not getattr(args, "allow_scope_filtering", False):
            print(
                "ERROR: Dataset includes rows outside the declared training scope. "
                "Export one JSONL per scope, or re-run with --allow-scope-filtering to drop mismatches explicitly."
            )
            print(
                f"       Declared scope: mold_id={scope_mold_id!r}  material_id={scope_material_id!r}  "
                f"machine_id={scope_machine_id!r}"
            )
            print(f"       Rows outside scope: {dropped}")
            return 2
        rows = matching_rows
        if dropped:
            print(
                f"Scope filter: dropped {dropped} rows not matching "
                f"mold_id={scope_mold_id!r} / material_id={scope_material_id!r} / "
                f"machine_id={scope_machine_id!r}"
            )
        if not rows:
            print(
                "ERROR: No rows remain after scope filtering. "
                "Check --mold-id / --material-id / --machine-id values."
            )
            return 2

    # Warn if material_id is absent across all rows (data collection gap).
    null_material = sum(1 for r in rows if not r.get("material_id"))
    if null_material:
        print(
            f"WARNING: {null_material}/{len(rows)} rows have no material_id — "
            "consider passing --material-id to scope training correctly."
        )

    # Warn on heterogeneous HMI schemas — training on mixed machine types degrades model quality.
    homogeneity = check_schema_homogeneity(rows)
    if not homogeneity["homogeneous"]:
        layouts = homogeneity.get("hmi_layouts_seen", [])
        families = sorted({lay.get("machine_family") for lay in layouts if lay.get("machine_family")})
        print(
            f"WARNING: Dataset contains {homogeneity['n_schemas']} distinct HMI schemas "
            f"(machine families: {families}). "
            "Training on heterogeneous schemas produces a union-schema model with sparse features "
            "that may have degraded accuracy. "
            "Recommended: re-run with --machine-id <family> to train one model per machine family."
        )

    cfg = GbtTrainingConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        cv_folds=args.cv_folds,
        null_strategy=args.null_strategy,
        min_feature_presence_ratio=args.min_feature_presence_ratio,
    )

    eligible_count = sum(
        1 for r in rows if (r.get("eligibility") or {}).get("training_ready")
    )
    readiness = assess_training_readiness(eligible_count)
    print(
        f"Training readiness: {readiness['level']}  ·  {readiness['message']}"
    )
    print(f"Training on up to {len(rows)} rows  (cv_folds={cfg.cv_folds}, n_estimators={cfg.n_estimators}) ...")
    try:
        result = train_suggestion_models(rows, config=cfg)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: Training failed — {exc}")
        return 2

    if not result.targets:
        print("ERROR: No targets could be trained (insufficient eligible rows?).")
        return 2

    # Print CV metric table.
    print(f"\nTrained {len(result.targets)} target(s) on {result.n_eligible_rows} eligible rows"
          f"  ({len(result.feature_keys)} features):\n")
    col_w = max(len(n) for n in result.targets) + 2
    print(f"  {'Target':<{col_w}}  Metric       Mean      Std")
    print("  " + "-" * (col_w + 36))
    for name, tr in result.targets.items():
        print(f"  {name:<{col_w}}  {tr.cv_metric_name:<12} {tr.cv_metric_value:7.4f}  ±{tr.cv_metric_std:.4f}")

    print(f"\nSelected features after constant pruning ({len(result.feature_keys)}):")
    for key in result.feature_keys:
        print(f"  - {key}")

    print("\nUsed LightGBM features:")
    for name, tr in result.targets.items():
        print(f"  {name}:")
        for key in getattr(tr, "used_feature_keys", result.feature_keys):
            print(f"    - {key}")

    # Persist training result.
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "train_result.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)

    # Write a scope.json alongside so predictive bundle can pick it up automatically.
    import json
    scope_path = output_dir / "scope.json"
    scope_path.write_text(
        json.dumps(
            {
                "mold_id": scope_mold_id,
                "material_id": scope_material_id,
                "machine_id": scope_machine_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nArtifacts written to: {output_dir}")
    print(f"  {pkl_path.name}")
    if scope_mold_id or scope_material_id or scope_machine_id:
        print(
            f"  scope.json  (mold_id={scope_mold_id!r}, "
            f"material_id={scope_material_id!r}, machine_id={scope_machine_id!r})"
        )
    return 0


def _default_predictive_train_output_dir(
    input_path: Path,
    *,
    mold_id: str | None,
    material_id: str | None,
    machine_id: str | None,
) -> Path:
    root = appconfig.get_predictive_runs_root()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [
        timestamp,
        _slugify_predictive_component(input_path.stem or "training_rows"),
    ]
    if mold_id:
        parts.append(f"mold-{_slugify_predictive_component(mold_id)}")
    if material_id:
        parts.append(f"material-{_slugify_predictive_component(material_id)}")
    if machine_id:
        parts.append(f"machine-{_slugify_predictive_component(machine_id)}")
    return root / "__".join(parts)


def _slugify_predictive_component(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value))
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return collapsed or "run"


def _handle_predictive_bundle(args) -> int:
    """Handle ``predictive bundle``.

    Loads the pickled ``TrainResult`` produced by ``predictive train``, exports
    all targets to ONNX, writes a ``manifest.json`` + ``training_meta.json``,
    and optionally packs the directory into a ``.sugbundle`` zip archive.
    """
    import json
    import pickle

    from .predictive.suggestion_bundle import pack_sugbundle, write_suggestion_bundle

    train_dir = resolve_path(args.train_dir)
    pkl_path = train_dir / "train_result.pkl"

    if not pkl_path.exists():
        print(f"ERROR: train_result.pkl not found in {train_dir}")
        print("       Run 'predictive train' first.")
        return 2

    # Load scope from scope.json written by 'predictive train', overridden by explicit flags.
    scope_mold_id: str | None = getattr(args, "mold_id", None)
    scope_material_id: str | None = getattr(args, "material_id", None)
    scope_machine_id: str | None = getattr(args, "machine_id", None)
    scope_path = train_dir / "scope.json"
    if scope_path.exists():
        saved_scope = json.loads(scope_path.read_text(encoding="utf-8"))
        if scope_mold_id is None:
            scope_mold_id = saved_scope.get("mold_id")
        if scope_material_id is None:
            scope_material_id = saved_scope.get("material_id")
        if scope_machine_id is None:
            scope_machine_id = saved_scope.get("machine_id")

    if not scope_mold_id or not scope_material_id:
        print("WARNING: Bundle scope is incomplete — mold_id or material_id is not set.")
        print("         The bundle will load with unscoped fallback in MoldPilot.")
        print("         Provide --mold-id and --material-id for production bundles.")

    print(f"Loading training result from {pkl_path} ...")
    with open(pkl_path, "rb") as fh:
        result = pickle.load(fh)

    output_dir = train_dir / "deploy"
    print(f"Writing bundle to {output_dir} ...")
    try:
        bundle_dir = write_suggestion_bundle(
            output_dir=output_dir,
            train_result=result,
            model_name=args.model_name,
            model_version=args.model_version,
            channel=args.channel,
            supersedes=args.supersedes,
            mold_id=scope_mold_id,
            material_id=scope_material_id,
            machine_id=scope_machine_id,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: Bundle write failed — {exc}")
        return 2

    print(f"  Bundle directory: {bundle_dir}")
    if scope_mold_id or scope_material_id or scope_machine_id:
        print(
            f"  Scope: mold_id={scope_mold_id!r}  material_id={scope_material_id!r}  "
            f"machine_id={scope_machine_id!r}"
        )

    if args.sugbundle:
        archive = pack_sugbundle(bundle_dir)
        print(f"  Packed archive:   {archive}")

    if getattr(args, "publish", False):
        from moldvision.publish import publish_bundle

        try:
            publish_result = publish_bundle(
                bundle_dir,
                role="startup_suggestion",
                channel=args.channel,
                dry_run=getattr(args, "publish_dry_run", False),
            )
        except Exception as exc:
            print(f"ERROR: Publishing failed — {exc}")
            return 2

        if getattr(args, "publish_dry_run", False):
            print("  Publish dry-run:")
        else:
            print("  Published:")
        print(json.dumps(publish_result, indent=2))

    print("\nDone.")
    return 0


# ── publish ────────────────────────────────────────────────────────────────


def handle_publish(args) -> int:
    """Handle the 'publish' subcommand."""
    from pathlib import Path
    from moldvision.publish import publish_bundle

    bundle_path = Path(args.bundle_path)
    if not bundle_path.exists():
        print(f"ERROR: Bundle path does not exist: {bundle_path}")
        return 1

    try:
        result = publish_bundle(
            bundle_path,
            role=args.role,
            channel=args.channel,
            compatible_layouts=args.compatible_layouts,
            dry_run=args.dry_run,
        )
    except ImportError as e:
        print(f"ERROR: {e}")
        return 1
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Publishing failed: {e}")
        return 1

    if args.dry_run:
        print("DRY RUN — catalog entry:")
    else:
        print("Published successfully:")

    import json
    print(json.dumps(result, indent=2))
    return 0
