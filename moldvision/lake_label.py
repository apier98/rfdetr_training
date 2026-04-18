"""Label-batch workflow for the ARIA Data Lake.

``label_batch_create``    — selects a subset of raw frames and copies them into
                            ``label_batches/<batch_id>/images/`` for Label Studio.

``label_batch_commit``    — reads the COCO export from Label Studio, merges
                            annotations back into the session annotation files,
                            and updates ``image_index.jsonl``.

``session_mark_backgrounds`` — marks all currently-unlabeled frames in a session
                            as ``background`` status in the index, so coverage
                            statistics correctly reflect that these images were
                            reviewed and confirmed to contain no objects of interest.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import uuid as _uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .lake import (
    DETECT_CLASSES,
    LABEL_STATUS_BACKGROUND,
    LABEL_STATUS_LABELED,
    LABEL_STATUS_UNLABELED,
    SEG_CLASSES,
    LakeConfig,
    filter_index,
    load_index,
    patch_index_records,
    save_index,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _batch_id(name: Optional[str]) -> str:
    short = _uuid.uuid4().hex[:8]
    if name:
        # sanitise: replace spaces/special chars with dashes
        safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in name)
        return f"{safe}-{short}"
    return f"batch-{short}"


def _batch_rel(batch_id: str) -> str:
    return f"label_batches/{batch_id}"


def _parse_coco(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────────
# Frame selection
# ──────────────────────────────────────────────────────────────────────────────

def _select_frames_random(
    frames: List[Dict[str, Any]],
    n: int,
    seed: int,
    min_frame_gap: int,
    skip_first: int,
    skip_last: int,
) -> List[Dict[str, Any]]:
    """Uniform random selection respecting skip and gap rules."""
    frames = sorted(frames, key=lambda r: r.get("frame_idx", 0))
    # Apply skip-first / skip-last
    if skip_first:
        frames = frames[skip_first:]
    if skip_last:
        frames = frames[:len(frames) - skip_last] if skip_last < len(frames) else []
    if not frames:
        return []

    if min_frame_gap <= 1:
        rng = random.Random(seed)
        rng.shuffle(frames)
        return frames[:n]

    # Enforce min_frame_gap: after choosing each frame, mask out following gap frames
    rng = random.Random(seed)
    pool = list(frames)
    rng.shuffle(pool)
    selected: List[Dict[str, Any]] = []
    excluded_until: Dict[int, bool] = {}
    for f in pool:
        fi = f.get("frame_idx", 0)
        if fi in excluded_until:
            continue
        selected.append(f)
        for g in range(fi, fi + min_frame_gap):
            excluded_until[g] = True
        if len(selected) >= n:
            break
    return selected


def _select_frames_temporal(
    frames: List[Dict[str, Any]],
    n: int,
    seed: int,
    min_frame_gap: int,
    skip_first: int,
    skip_last: int,
) -> List[Dict[str, Any]]:
    """Evenly-spaced selection across the session timeline."""
    frames = sorted(frames, key=lambda r: r.get("frame_idx", 0))
    if skip_first:
        frames = frames[skip_first:]
    if skip_last:
        frames = frames[:len(frames) - skip_last] if skip_last < len(frames) else []
    if not frames:
        return []

    if n >= len(frames):
        return frames

    # Compute ideal step
    step = max(1, len(frames) // n)
    step = max(step, min_frame_gap)

    candidates: List[Dict[str, Any]] = frames[::step]
    if len(candidates) > n:
        rng = random.Random(seed)
        candidates = rng.sample(candidates, n)
        candidates.sort(key=lambda r: r.get("frame_idx", 0))
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# label_batch_create
# ──────────────────────────────────────────────────────────────────────────────

def label_batch_create(
    cfg: LakeConfig,
    *,
    task: str,
    sessions: Optional[List[str]] = None,
    all_sessions: bool = False,
    machine_id: Optional[str] = None,
    mold_id: Optional[str] = None,
    marker: Optional[str] = None,
    only_unlabeled: bool = True,
    n: int = 200,
    sample_mode: str = "random",
    min_frame_gap: int = 1,
    skip_first: int = 0,
    skip_last: int = 0,
    seed: int = 42,
    batch_name: Optional[str] = None,
) -> str:
    """Select frames and create a label batch.

    Returns the ``batch_id`` of the newly created batch.
    """
    records = load_index(cfg.root)

    label_status_filter = LABEL_STATUS_UNLABELED if only_unlabeled else None
    session_ids = sessions if (sessions and not all_sessions) else None

    candidates = filter_index(
        records,
        task=task,
        session_ids=session_ids,
        label_status=label_status_filter,
        machine_id=machine_id,
        mold_id=mold_id,
        marker=marker,
    )

    if not candidates:
        raise RuntimeError("No matching unlabeled frames found. Check your filters or run 'lake session import' first.")

    # Group by session and distribute n proportionally
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in candidates:
        by_session[r["session_id"]].append(r)

    total_available = sum(len(v) for v in by_session.values())
    selected_frames: List[Dict[str, Any]] = []

    select_fn = _select_frames_temporal if sample_mode == "temporal" else _select_frames_random

    # Proportional allocation across sessions (Largest Remainder Method)
    session_ids_sorted = sorted(by_session.keys())
    shares = {sid: n * len(by_session[sid]) / total_available for sid in session_ids_sorted}
    floors = {sid: int(s) for sid, s in shares.items()}
    remainder = n - sum(floors.values())
    remainders_sorted = sorted(session_ids_sorted, key=lambda sid: shares[sid] - floors[sid], reverse=True)
    allocation: Dict[str, int] = dict(floors)
    for i in range(remainder):
        allocation[remainders_sorted[i]] += 1

    session_counts: Dict[str, int] = {}
    for sid in session_ids_sorted:
        alloc = allocation[sid]
        if alloc == 0:
            session_counts[sid] = 0
            continue
        # Use different seed per session (deterministic but session-specific)
        s_seed = seed ^ hash(sid) & 0xFFFFFFFF
        chosen = select_fn(
            by_session[sid],
            alloc,
            s_seed,
            min_frame_gap=min_frame_gap,
            skip_first=skip_first,
            skip_last=skip_last,
        )
        selected_frames.extend(chosen)
        session_counts[sid] = len(chosen)

    if not selected_frames:
        raise RuntimeError("Frame selection returned 0 images. Try loosening filters or increasing --n.")

    bid = _batch_id(batch_name)
    batch_rel = _batch_rel(bid)
    storage = cfg.storage()
    storage.makedirs(f"{batch_rel}/images")
    storage.makedirs(f"{batch_rel}/export")

    # Copy frames into batch/images/
    for r in selected_frames:
        src_abs = storage.abs_path(r["rel_path"])
        dst_rel = f"{batch_rel}/images/{Path(r['rel_path']).name}"
        storage.copy_in(src_abs, dst_rel, overwrite=True)

    frame_rel_paths = [r["rel_path"] for r in selected_frames]

    # Write batch_meta.json
    batch_meta = {
        "batch_id":     bid,
        "task":         task,
        "status":       "open",
        "created_at":   datetime.utcnow().isoformat() + "Z",
        "committed_at": None,
        "selection": {
            "mode":          sample_mode,
            "n":             n,
            "min_frame_gap": min_frame_gap,
            "skip_first":    skip_first,
            "skip_last":     skip_last,
            "seed":          seed,
            "sessions":      sessions or list(session_ids_sorted),
            "filters": {
                k: v for k, v in {
                    "machine_id": machine_id,
                    "mold_id":    mold_id,
                    "marker":     marker,
                }.items() if v is not None
            },
        },
        "frames":        frame_rel_paths,
        "commit_summary": None,
    }
    storage.write_text(f"{batch_rel}/batch_meta.json", json.dumps(batch_meta, indent=2, ensure_ascii=False))

    # Print instructions
    classes = DETECT_CLASSES if task == "detect" else SEG_CLASSES
    ann_type = "RectangleLabels" if task == "detect" else "PolygonLabels"

    print(f"\nBatch created: {storage.abs_path(batch_rel)}")
    print(f"Images:        {len(selected_frames)}")
    per_sess = "  +  ".join(f"{sid}={cnt}" for sid, cnt in session_counts.items() if cnt > 0)
    print(f"Sessions:      {per_sess}")
    print(f"Sample mode:   {sample_mode}  |  min-frame-gap: {min_frame_gap}  |  seed: {seed}")
    print(f"\nNext steps:")
    if task == "detect":
        print(f"  1. [Optional] Start ML pre-labeling backend:")
        print(f"       moldvision label-studio-backend --task {task}")
    print(f"  2. In Label Studio: create project → import images from:")
    print(f"       {storage.abs_path(batch_rel + '/images')}")
    print(f"     Annotation type: {ann_type}")
    print(f"     Classes: {', '.join(classes)}")
    print(f"  3. Export COCO JSON to:")
    print(f"       {storage.abs_path(batch_rel + '/export')}/")
    print(f"  4. Commit:")
    print(f"       moldvision lake label-batch commit --batch {bid}")

    return bid


# ──────────────────────────────────────────────────────────────────────────────
# label_batch_commit
# ──────────────────────────────────────────────────────────────────────────────

def _find_export_coco(batch_abs: Path, explicit: Optional[Path]) -> Path:
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(f"--coco-json not found: {explicit}")
        return explicit
    export_dir = batch_abs / "export"
    if not export_dir.exists():
        raise FileNotFoundError(f"No export/ folder found in batch: {batch_abs}")
    jsons = list(export_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No JSON export found in {export_dir}. Did you export from Label Studio?")
    if len(jsons) > 1:
        # Prefer _annotations.coco.json if present
        named = [j for j in jsons if "_annotations" in j.name]
        if named:
            return named[0]
    return jsons[0]


def _merge_coco_into_session_file(
    session_coco_path: Path,
    new_images: List[Dict[str, Any]],
    new_anns: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    batch_id: str,
) -> None:
    """Merge *new_images* / *new_anns* into the persistent session COCO file.

    Creates the file if it doesn't exist. Deduplicates by image file_name.
    """
    if session_coco_path.exists():
        try:
            existing = json.loads(session_coco_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {"images": [], "annotations": [], "categories": categories}
    else:
        session_coco_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {"images": [], "annotations": [], "categories": categories}

    existing_fnames = {im["file_name"] for im in existing.get("images", [])}
    next_img_id = max((im["id"] for im in existing.get("images", [])), default=0) + 1
    next_ann_id = max((a["id"] for a in existing.get("annotations", [])), default=0) + 1

    img_id_map: Dict[int, int] = {}
    for im in new_images:
        if im["file_name"] in existing_fnames:
            continue
        new_im = dict(im)
        old_id = new_im["id"]
        new_im["id"] = next_img_id
        new_im.setdefault("extra", {})["label_batch_id"] = batch_id
        img_id_map[old_id] = next_img_id
        existing["images"].append(new_im)
        next_img_id += 1

    for ann in new_anns:
        old_img_id = ann["image_id"]
        if old_img_id not in img_id_map:
            continue
        new_ann = dict(ann)
        new_ann["image_id"] = img_id_map[old_img_id]
        new_ann["id"] = next_ann_id
        existing["annotations"].append(new_ann)
        next_ann_id += 1

    # Ensure categories are set (use first batch's categories if not present)
    if not existing.get("categories"):
        existing["categories"] = categories

    session_coco_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def label_batch_commit(
    cfg: LakeConfig,
    *,
    batch_id: str,
    coco_json_path: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Commit a labeled batch back into the data lake.

    Returns a summary dict with per-session and per-class counts.
    """
    storage = cfg.storage()
    batch_rel = _batch_rel(batch_id)
    batch_abs = storage.abs_path(batch_rel)

    meta_path = batch_abs / "batch_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Batch not found: {batch_abs}")
    batch_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if batch_meta.get("status") == "committed" and not dry_run:
        print(f"Warning: batch '{batch_id}' already committed. Re-committing will overwrite annotations.")

    task: str = batch_meta["task"]
    batch_frames: List[str] = batch_meta.get("frames", [])

    # Build frame filename → rel_path map from batch metadata
    fname_to_rel: Dict[str, str] = {Path(rp).name: rp for rp in batch_frames}

    def _candidate_export_names(raw_file_name: str) -> List[str]:
        """Return possible filenames that may map an export image to batch frames.

        Label Studio COCO exports may store file_name like:
          ../../label-studio/media/upload/14/3efaed63-my_image.jpg
        while our batch stores:
          my_image.jpg
        """
        base = Path(raw_file_name).name
        out = [base]
        if "-" in base:
            prefix, rest = base.split("-", 1)
            # Common Label Studio upload prefix is 8 hex chars.
            if len(prefix) == 8 and all(c in "0123456789abcdefABCDEF" for c in prefix):
                out.append(rest)
        return out

    # Find and parse COCO export
    coco_path = _find_export_coco(batch_abs, coco_json_path)
    coco = _parse_coco(coco_path)

    export_images: List[Dict[str, Any]] = coco.get("images", [])
    export_anns: List[Dict[str, Any]] = coco.get("annotations", [])
    export_cats: List[Dict[str, Any]] = coco.get("categories", [])

    # Group by session
    img_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    ann_by_img_id: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in export_anns:
        ann_by_img_id[int(ann["image_id"])].append(ann)

    committed_rel_paths: List[str] = []
    skipped_count = 0

    for im in export_images:
        rel_path = None
        for candidate in _candidate_export_names(im.get("file_name", "")):
            rel_path = fname_to_rel.get(candidate)
            if rel_path:
                break
        if not rel_path:
            skipped_count += 1
            continue
        # Extract session from rel_path: sessions/<session_id>/...
        parts = Path(rel_path).parts
        if len(parts) >= 2 and parts[0] == "sessions":
            sid = parts[1]
        else:
            skipped_count += 1
            continue
        img_by_session[sid].append(im)
        committed_rel_paths.append(rel_path)

    # Per-class annotation counter
    from collections import Counter
    class_map = {c["id"]: c["name"] for c in export_cats}
    ann_counter: Counter = Counter()

    if not dry_run:
        ann_task_dir = "detect" if task == "detect" else "seg"
        for sid, images in img_by_session.items():
            img_ids_in_session = {im["id"] for im in images}
            anns_for_session = [a for a in export_anns if a.get("image_id") in img_ids_in_session]
            for a in anns_for_session:
                ann_counter[class_map.get(int(a.get("category_id", -1)), str(a.get("category_id", "?")))] += 1

            session_coco_path = storage.abs_path(
                f"sessions/{sid}/annotations/{ann_task_dir}/_annotations.coco.json"
            )
            _merge_coco_into_session_file(session_coco_path, images, anns_for_session, export_cats, batch_id)

        # Update image_index.jsonl
        status_field = "detect_status" if task == "detect" else "seg_status"
        batch_id_field = "detect_batch_id" if task == "detect" else "seg_batch_id"
        patch_index_records(cfg.root, committed_rel_paths, {
            status_field:   LABEL_STATUS_LABELED,
            batch_id_field: batch_id,
        })

        # Update batch_meta.json
        batch_meta["status"] = "committed"
        batch_meta["committed_at"] = datetime.utcnow().isoformat() + "Z"
        batch_meta["commit_summary"] = {
            "images_committed":           len(committed_rel_paths),
            "images_skipped_not_in_export": skipped_count,
            "annotations_per_class":      dict(ann_counter),
        }
        meta_path.write_text(json.dumps(batch_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}Batch commit: {batch_id}")
    print(f"  Images committed:    {len(committed_rel_paths)}")
    print(f"  Images skipped:      {skipped_count}")
    if not dry_run:
        print(f"  Annotations per class:")
        for cls_name, cnt in sorted(ann_counter.items()):
            print(f"    {cls_name}: {cnt}")

    return {
        "images_committed": len(committed_rel_paths),
        "images_skipped":   skipped_count,
        "annotations_per_class": dict(ann_counter),
    }


# ──────────────────────────────────────────────────────────────────────────────
# label_batch_status  (list open/committed batches)
# ──────────────────────────────────────────────────────────────────────────────

def label_batch_status(cfg: LakeConfig, task: Optional[str] = None) -> None:
    """Print all label batches with their status."""
    root = cfg.storage().abs_path("label_batches")
    if not root.exists():
        print("No label batches found.")
        return

    header = f"{'batch_id':<45} {'task':<8} {'status':<12} {'images':>7} {'created_at':<22}"
    print(header)
    print("─" * len(header))

    for bd in sorted(root.iterdir()):
        if not bd.is_dir():
            continue
        meta_path = bd / "batch_meta.json"
        if not meta_path.exists():
            continue
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if task and m.get("task") != task:
            continue
        n_frames = len(m.get("frames", []))
        print(f"{m.get('batch_id','?'):<45} {m.get('task','?'):<8} {m.get('status','?'):<12} {n_frames:>7} {m.get('created_at',''):<22}")


# ──────────────────────────────────────────────────────────────────────────────
# session_mark_backgrounds
# ──────────────────────────────────────────────────────────────────────────────

def session_mark_backgrounds(
    cfg: LakeConfig,
    *,
    session_id: str,
    task: str,
    dry_run: bool = False,
) -> int:
    """Mark all currently-unlabeled frames in *session_id* as ``background``.

    This is useful when you have inspected a session and confirmed that the
    remaining unlabeled frames contain no objects of interest (pure negatives).
    The index is updated so that coverage statistics correctly show 100% and
    the images are excluded from future label-batch creation by default.

    Returns the number of frames updated.
    """
    records = load_index(cfg.root)

    unlabeled = filter_index(
        records,
        task=task,
        session_ids=[session_id],
        label_status=LABEL_STATUS_UNLABELED,
    )

    if not unlabeled:
        print(f"No unlabeled {task} frames found in session '{session_id}'.")
        return 0

    rel_paths = [r["rel_path"] for r in unlabeled]

    if dry_run:
        print(f"[DRY RUN] Would mark {len(rel_paths)} frame(s) as background in session '{session_id}' (task={task}).")
        for rp in rel_paths[:10]:
            print(f"  {rp}")
        if len(rel_paths) > 10:
            print(f"  … and {len(rel_paths) - 10} more")
        return len(rel_paths)

    status_field = "detect_status" if task == "detect" else "seg_status"
    patch_index_records(cfg.root, rel_paths, {status_field: LABEL_STATUS_BACKGROUND})

    print(f"Marked {len(rel_paths)} frame(s) as background in session '{session_id}' (task={task}).")
    return len(rel_paths)
