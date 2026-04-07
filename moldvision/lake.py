"""Core ARIA Data Lake logic.

Responsibilities:
- ``LakeConfig``       — load / save ``data_lake_config.json``; resolve lake root.
- ``init_lake``        — create the folder skeleton and write config.
- ``session_import``   — copy raw frames, write ``session_meta.json``, update index.
- ``index_rebuild``    — scan all sessions and rewrite ``image_index.jsonl``.
- ``index_stats``      — print per-session and per-class coverage.
- ``session_list``     — query index and print a coverage table.
- ``index helpers``    — load, filter, append, and patch individual index records.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .lake_storage import LocalLakeStorage, make_storage

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ENV_LAKE_ROOT = "ARIA_DATA_LAKE"
CONFIG_FILENAME = "data_lake_config.json"
INDEX_FILENAME = "image_index.jsonl"

_DEFAULT_ROOT_WIN = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ARIA" / "DataLake"
_DEFAULT_ROOT_UNIX = Path.home() / ".aria" / "data_lake"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

DETECT_CLASSES = ["Component_Base", "Weld_Line", "Sink_Mark", "Flash", "Burn_Mark"]
SEG_CLASSES = ["HMI_Screen"]

# Valid label-status values in image_index.jsonl
LABEL_STATUS_UNLABELED = "unlabeled"
LABEL_STATUS_LABELED = "labeled"
LABEL_STATUS_HARD_NEGATIVE = "hard_negative"
LABEL_STATUS_BACKGROUND = "background"
LABEL_STATUS_NA = "n/a"


# ──────────────────────────────────────────────────────────────────────────────
# Lake configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LakeConfig:
    root: Path
    storage_backend: str = "local"

    # extra fields from config file are stored here
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def default_root() -> Path:
        env = os.environ.get(ENV_LAKE_ROOT)
        if env:
            return Path(env)
        if sys.platform == "win32":
            return _DEFAULT_ROOT_WIN
        return _DEFAULT_ROOT_UNIX

    @staticmethod
    def find(override: Optional[Path] = None) -> "LakeConfig":
        """Load config from *override* root, env var, or default location."""
        root = override or LakeConfig.default_root()
        cfg_path = root / CONFIG_FILENAME
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            return LakeConfig(
                root=Path(data.get("root", str(root))),
                storage_backend=data.get("storage_backend", "local"),
                extra={k: v for k, v in data.items() if k not in ("root", "storage_backend")},
            )
        return LakeConfig(root=root)

    def save(self) -> None:
        cfg_path = self.root / CONFIG_FILENAME
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"root": str(self.root), "storage_backend": self.storage_backend, **self.extra}
        cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def storage(self) -> LocalLakeStorage:
        return make_storage(self.root)


# ──────────────────────────────────────────────────────────────────────────────
# Lake initialisation
# ──────────────────────────────────────────────────────────────────────────────

_SKELETON_DIRS = [
    "sessions",
    "label_batches",
    "pools/hard_negatives",
    "pools/backgrounds",
    "datasets",
    "models/defect_detection",
    "models/monitor_segmentation",
]


def init_lake(root: Path) -> LakeConfig:
    """Create the folder skeleton and write ``data_lake_config.json``.

    Safe to call on an existing lake — already-present dirs are left intact.
    """
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    for rel in _SKELETON_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)

    # Seed empty pool manifests
    for pool in ("hard_negatives", "backgrounds"):
        mf = root / "pools" / pool / "manifest.jsonl"
        if not mf.exists():
            mf.write_text("", encoding="utf-8")

    cfg = LakeConfig(root=root)
    cfg.save()
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# image_index.jsonl helpers
# ──────────────────────────────────────────────────────────────────────────────

def _index_path(root: Path) -> Path:
    return root / INDEX_FILENAME


def load_index(root: Path) -> List[Dict[str, Any]]:
    """Load all records from ``image_index.jsonl``.  Returns ``[]`` if missing."""
    p = _index_path(root)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def save_index(root: Path, records: List[Dict[str, Any]]) -> None:
    """Overwrite ``image_index.jsonl`` with *records*."""
    _index_path(root).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else ""),
        encoding="utf-8",
    )


def append_index_records(root: Path, new_records: List[Dict[str, Any]]) -> None:
    """Append *new_records* to ``image_index.jsonl``."""
    if not new_records:
        return
    p = _index_path(root)
    with p.open("a", encoding="utf-8") as fh:
        for r in new_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def patch_index_record(root: Path, rel_path: str, updates: Dict[str, Any]) -> bool:
    """Update fields in the record for *rel_path*.  Returns True if found."""
    records = load_index(root)
    found = False
    for r in records:
        if r.get("rel_path") == rel_path:
            r.update(updates)
            found = True
    if found:
        save_index(root, records)
    return found


def patch_index_records(root: Path, rel_paths: List[str], updates: Dict[str, Any]) -> int:
    """Batch-update fields for all records in *rel_paths*. Returns count updated."""
    path_set = set(rel_paths)
    records = load_index(root)
    count = 0
    for r in records:
        if r.get("rel_path") in path_set:
            r.update(updates)
            count += 1
    save_index(root, records)
    return count


def filter_index(
    records: List[Dict[str, Any]],
    *,
    task: Optional[str] = None,
    session_ids: Optional[List[str]] = None,
    label_status: Optional[str] = None,
    machine_id: Optional[str] = None,
    mold_id: Optional[str] = None,
    part_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    marker: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return index records matching all supplied filters (AND logic)."""
    frame_type = {"detect": "inspection", "seg": "monitor"}.get(task or "", None)
    out = []
    for r in records:
        if frame_type and r.get("frame_type") != frame_type:
            continue
        if session_ids and r.get("session_id") not in session_ids:
            continue
        if machine_id and r.get("machine_id") != machine_id:
            continue
        if mold_id and r.get("mold_id") != mold_id:
            continue
        if part_id and r.get("part_id") != part_id:
            continue
        if from_date and (r.get("started_at") or "") < from_date:
            continue
        if to_date and (r.get("started_at") or "") > to_date:
            continue
        if marker:
            markers: List[str] = r.get("markers") or []
            if not any(marker.lower() in m.lower() for m in markers):
                continue
        # label_status filter for the relevant task
        if label_status:
            if task == "detect":
                if r.get("detect_status") != label_status:
                    continue
            elif task == "seg":
                if r.get("seg_status") != label_status:
                    continue
            else:
                # no task filter: match either
                if r.get("detect_status") != label_status and r.get("seg_status") != label_status:
                    continue
        out.append(r)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Session metadata helpers
# ──────────────────────────────────────────────────────────────────────────────

def _meta_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the fields we store in image_index records from a session manifest."""
    return {
        "session_id":     manifest.get("session_id", ""),
        "machine_id":     manifest.get("machine_id", ""),
        "mold_id":        manifest.get("mold_id", ""),
        "part_id":        manifest.get("part_id", ""),
        "operator_name":  manifest.get("operator_name", ""),
        "batch_number":   manifest.get("batch_number", ""),
        "started_at":     manifest.get("started_at", ""),
        "markers":        manifest.get("markers") or [],
    }


def _make_index_record(
    rel_path: str,
    meta: Dict[str, Any],
    frame_type: str,  # "inspection" or "monitor"
    frame_idx: int,
) -> Dict[str, Any]:
    task_na = "n/a"
    detect_status = LABEL_STATUS_UNLABELED if frame_type == "inspection" else task_na
    seg_status = LABEL_STATUS_UNLABELED if frame_type == "monitor" else task_na
    return {
        "rel_path":       rel_path,
        "session_id":     meta.get("session_id", ""),
        "machine_id":     meta.get("machine_id", ""),
        "mold_id":        meta.get("mold_id", ""),
        "part_id":        meta.get("part_id", ""),
        "operator_name":  meta.get("operator_name", ""),
        "batch_number":   meta.get("batch_number", ""),
        "started_at":     meta.get("started_at", ""),
        "markers":        meta.get("markers") or [],
        "frame_type":     frame_type,
        "frame_idx":      frame_idx,
        "detect_status":  detect_status,
        "detect_batch_id": None,
        "seg_status":     seg_status,
        "seg_batch_id":   None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Session import
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionImportResult:
    session_id: str
    inspection_frames_added: int
    monitor_frames_added: int
    already_existed: bool


def session_import(
    cfg: LakeConfig,
    *,
    session_meta_path: Path,
    inspection_frames_dir: Optional[Path],
    monitor_frames_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> SessionImportResult:
    """Import a qual session into the data lake.

    Copies frames, writes ``session_meta.json``, appends ``image_index.jsonl``.
    Raises ``FileExistsError`` if session already exists and *overwrite* is False.
    """
    manifest = json.loads(session_meta_path.read_text(encoding="utf-8"))
    session_id: str = manifest.get("session_id", "")
    if not session_id:
        raise ValueError("session_meta.json must contain a non-empty 'session_id' field")

    storage = cfg.storage()
    session_rel = f"sessions/{session_id}"
    meta_rel = f"{session_rel}/session_meta.json"

    already_existed = storage.exists(meta_rel)
    if already_existed and not overwrite:
        raise FileExistsError(
            f"Session '{session_id}' already exists in the lake. Use --overwrite to replace."
        )

    # Write session_meta.json
    storage.write_text(meta_rel, json.dumps(manifest, indent=2, ensure_ascii=False))

    meta = _meta_from_manifest(manifest)
    existing_paths = {r["rel_path"] for r in load_index(cfg.root)}
    new_records: List[Dict[str, Any]] = []

    def _import_frames(src_dir: Path, frame_type: str) -> int:
        subdir = "inspection_frames" if frame_type == "inspection" else "monitor_frames"
        count = 0
        frames = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        for idx, frame_path in enumerate(frames):
            rel = f"{session_rel}/{subdir}/{frame_path.name}"
            if rel not in existing_paths:
                storage.copy_in(frame_path, rel, overwrite=overwrite)
                new_records.append(_make_index_record(rel, meta, frame_type, idx))
                count += 1
            elif overwrite:
                storage.copy_in(frame_path, rel, overwrite=True)
        return count

    insp_count = 0
    mon_count = 0

    if inspection_frames_dir and inspection_frames_dir.exists():
        insp_count = _import_frames(inspection_frames_dir, "inspection")
    if monitor_frames_dir and monitor_frames_dir.exists():
        mon_count = _import_frames(monitor_frames_dir, "monitor")

    append_index_records(cfg.root, new_records)

    return SessionImportResult(
        session_id=session_id,
        inspection_frames_added=insp_count,
        monitor_frames_added=mon_count,
        already_existed=already_existed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# External data import  (pre-labeled or unlabeled data from outside MoldPilot)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExternalImportResult:
    session_id: str
    images_added: int
    images_labeled: int
    images_unlabeled: int
    already_existed: bool


def external_import(
    cfg: LakeConfig,
    *,
    images_dir: Path,
    task: str,
    coco_json: Optional[Path] = None,
    session_id: Optional[str] = None,
    name: Optional[str] = None,
    machine_id: Optional[str] = None,
    mold_id: Optional[str] = None,
    part_id: Optional[str] = None,
    notes: Optional[str] = None,
    overwrite: bool = False,
) -> ExternalImportResult:
    """Import externally-sourced images (and optional pre-existing annotations)
    into the data lake as a synthetic session.

    The imported data becomes a fully first-class session: it participates in
    ``lake pull``, ``--max-per-session`` distribution rules, and the full
    traceability chain identically to data that came from MoldPilot.

    Parameters
    ----------
    images_dir:
        Directory of JPEG/PNG images to import.
    task:
        ``"detect"`` (inspection frames) or ``"seg"`` (monitor frames).
    coco_json:
        Optional COCO annotation file.  Only images that appear in this file
        will be marked ``labeled``; images without annotations remain
        ``unlabeled`` (partial annotation is expected and supported).
    session_id:
        Custom session ID.  Defaults to ``external_<timestamp>_<short_uuid>``.
    name, machine_id, mold_id, part_id, notes:
        Metadata written into ``session_meta.json``.
    overwrite:
        Replace an existing session with the same ID.
    """
    import uuid as _uuid

    # Generate a session ID that is clearly external-origin
    if not session_id:
        short = _uuid.uuid4().hex[:8]
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        session_id = f"external_{ts}_{short}"

    storage = cfg.storage()
    frame_type = "inspection" if task == "detect" else "monitor"
    subdir = "inspection_frames" if frame_type == "inspection" else "monitor_frames"
    session_rel = f"sessions/{session_id}"
    meta_rel = f"{session_rel}/session_meta.json"

    already_existed = storage.exists(meta_rel)
    if already_existed and not overwrite:
        raise FileExistsError(
            f"Session '{session_id}' already exists. Use --overwrite to replace."
        )

    # Build and write session_meta.json
    manifest: Dict[str, Any] = {
        "session_id":    session_id,
        "source":        "external",
        "name":          name or session_id,
        "machine_id":    machine_id or "",
        "mold_id":       mold_id or "",
        "part_id":       part_id or "",
        "operator_name": "",
        "batch_number":  "",
        "started_at":    datetime.utcnow().isoformat() + "Z",
        "ended_at":      "",
        "status":        "completed",
        "markers":       [],
        "video_chunks":  [],
        "notes":         notes or "",
    }
    storage.write_text(meta_rel, json.dumps(manifest, indent=2, ensure_ascii=False))

    # Parse COCO to build a set of annotated file names
    annotated_fnames: set = set()
    if coco_json and coco_json.exists():
        try:
            coco_data = json.loads(coco_json.read_text(encoding="utf-8"))
            # Images that have at least one annotation
            ann_img_ids = {a.get("image_id") for a in coco_data.get("annotations", [])}
            for im in coco_data.get("images", []):
                if im.get("id") in ann_img_ids:
                    annotated_fnames.add(Path(im.get("file_name", "")).name)
        except Exception as e:
            raise ValueError(f"Could not parse COCO JSON at {coco_json}: {e}") from e

    # Copy images
    existing_paths = {r["rel_path"] for r in load_index(cfg.root)}
    new_records: List[Dict[str, Any]] = []
    meta_fields = _meta_from_manifest(manifest)

    frames = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    images_labeled = images_unlabeled = 0

    ann_task_dir = "detect" if task == "detect" else "seg"

    for idx, frame_path in enumerate(frames):
        rel = f"{session_rel}/{subdir}/{frame_path.name}"
        if rel not in existing_paths or overwrite:
            storage.copy_in(frame_path, rel, overwrite=overwrite)

        if rel not in existing_paths:
            rec = _make_index_record(rel, meta_fields, frame_type, idx)
            # If this image has annotations, mark it labeled immediately
            is_labeled = bool(annotated_fnames) and frame_path.name in annotated_fnames
            if is_labeled:
                status_field = "detect_status" if task == "detect" else "seg_status"
                rec[status_field] = LABEL_STATUS_LABELED
                images_labeled += 1
            else:
                images_unlabeled += 1
            new_records.append(rec)

    append_index_records(cfg.root, new_records)

    # Write COCO annotations into the session annotation folder
    if coco_json and coco_json.exists():
        ann_rel = f"{session_rel}/annotations/{ann_task_dir}/_annotations.coco.json"
        storage.makedirs(f"{session_rel}/annotations/{ann_task_dir}")
        storage.write_text(ann_rel, coco_json.read_text(encoding="utf-8"))

    return ExternalImportResult(
        session_id=session_id,
        images_added=len(new_records),
        images_labeled=images_labeled,
        images_unlabeled=images_unlabeled,
        already_existed=already_existed,
    )



def index_rebuild(cfg: LakeConfig) -> int:
    """Full scan of all sessions — rewrites ``image_index.jsonl`` from scratch.

    Preserves existing annotation status by merging with the old index.
    Returns number of records written.
    """
    root = cfg.root
    sessions_dir = root / "sessions"
    if not sessions_dir.exists():
        save_index(root, [])
        return 0

    old_index = {r["rel_path"]: r for r in load_index(root)}
    new_records: List[Dict[str, Any]] = []

    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        meta_path = session_dir / "session_meta.json"
        if meta_path.exists():
            try:
                manifest = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {"session_id": session_id}
        else:
            manifest = {"session_id": session_id}
        meta = _meta_from_manifest(manifest)

        for frame_type, subdir in (("inspection", "inspection_frames"), ("monitor", "monitor_frames")):
            frames_dir = session_dir / subdir
            if not frames_dir.exists():
                continue
            frames = sorted(p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            for idx, fp in enumerate(frames):
                rel = f"sessions/{session_id}/{subdir}/{fp.name}"
                if rel in old_index:
                    # preserve annotation status from old index
                    new_records.append(old_index[rel])
                else:
                    new_records.append(_make_index_record(rel, meta, frame_type, idx))

    save_index(root, new_records)
    return len(new_records)


# ──────────────────────────────────────────────────────────────────────────────
# index --stats
# ──────────────────────────────────────────────────────────────────────────────

def index_stats(cfg: LakeConfig, task: Optional[str] = None) -> None:
    """Print per-session coverage statistics."""
    records = load_index(cfg.root)
    if not records:
        print("image_index.jsonl is empty.")
        return

    from collections import defaultdict

    # Group by session
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_session[r.get("session_id", "?")].append(r)

    header = f"{'session_id':<44} {'machine':<10} {'mold':<12} {'raw':>6} {'detect':>8} {'seg':>6} {'coverage':>10}"
    print(header)
    print("─" * len(header))

    total_raw = total_detect = total_seg = 0
    for sid in sorted(by_session):
        recs = by_session[sid]
        machine = recs[0].get("machine_id", "")
        mold = recs[0].get("mold_id", "")

        insp = [r for r in recs if r.get("frame_type") == "inspection"]
        mon = [r for r in recs if r.get("frame_type") == "monitor"]

        detect_labeled = sum(1 for r in insp if r.get("detect_status") == LABEL_STATUS_LABELED)
        seg_labeled = sum(1 for r in mon if r.get("seg_status") == LABEL_STATUS_LABELED)

        raw_count = len(insp) if (task is None or task == "detect") else len(mon)
        labeled = detect_labeled if task == "detect" else seg_labeled if task == "seg" else detect_labeled + seg_labeled
        raw = len(insp) + len(mon) if task is None else raw_count

        coverage_pct = f"{(detect_labeled / len(insp) * 100):.0f}%" if insp else "—"
        if task == "seg":
            coverage_pct = f"{(seg_labeled / len(mon) * 100):.0f}%" if mon else "—"

        det_str = str(detect_labeled) if insp else "—"
        seg_str = str(seg_labeled) if mon else "—"
        print(f"{sid:<44} {machine:<10} {mold:<12} {len(insp) + len(mon):>6} {det_str:>8} {seg_str:>6} {coverage_pct:>10}")
        total_raw += len(insp) + len(mon)
        total_detect += detect_labeled
        total_seg += seg_labeled

    print("─" * len(header))
    print(f"{'TOTAL':<44} {'':<10} {'':<12} {total_raw:>6} {total_detect:>8} {total_seg:>6}")


# ──────────────────────────────────────────────────────────────────────────────
# session list  (table output; filtering via filter_index)
# ──────────────────────────────────────────────────────────────────────────────

def session_list(
    cfg: LakeConfig,
    *,
    machine_id: Optional[str] = None,
    mold_id: Optional[str] = None,
    part_id: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    task: Optional[str] = None,
    label_status: Optional[str] = None,
    marker: Optional[str] = None,
    min_frames: int = 0,
) -> List[str]:
    """Print a session table and return the matching session IDs."""
    from collections import defaultdict

    records = load_index(cfg.root)

    # Apply metadata filters on sessions (not on per-record level for listing)
    # We get all sessions first, then filter by session-level metadata
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_session[r.get("session_id", "?")].append(r)

    matching: List[str] = []

    header = f"{'session_id':<44} {'machine':<10} {'mold':<12} {'raw':>6} {'detect':>8} {'seg':>6} {'coverage':>10}"
    print(header)
    print("─" * len(header))

    for sid in sorted(by_session):
        recs = by_session[sid]
        first = recs[0]

        if machine_id and first.get("machine_id") != machine_id:
            continue
        if mold_id and first.get("mold_id") != mold_id:
            continue
        if part_id and first.get("part_id") != part_id:
            continue
        if from_date and (first.get("started_at") or "") < from_date:
            continue
        if to_date and (first.get("started_at") or "") > to_date:
            continue
        if marker:
            markers = first.get("markers") or []
            if not any(marker.lower() in m.lower() for m in markers):
                continue

        insp = [r for r in recs if r.get("frame_type") == "inspection"]
        mon = [r for r in recs if r.get("frame_type") == "monitor"]
        detect_labeled = sum(1 for r in insp if r.get("detect_status") == LABEL_STATUS_LABELED)
        seg_labeled = sum(1 for r in mon if r.get("seg_status") == LABEL_STATUS_LABELED)

        # task / label_status filter
        if task == "detect":
            if label_status == LABEL_STATUS_LABELED and detect_labeled == 0:
                continue
            if label_status == LABEL_STATUS_UNLABELED and detect_labeled == len(insp):
                continue
        elif task == "seg":
            if label_status == LABEL_STATUS_LABELED and seg_labeled == 0:
                continue
            if label_status == LABEL_STATUS_UNLABELED and seg_labeled == len(mon):
                continue

        total_raw = len(insp) + len(mon)
        if min_frames and total_raw < min_frames:
            continue

        coverage_pct = f"{(detect_labeled / len(insp) * 100):.0f}%" if insp else "—"
        if task == "seg":
            coverage_pct = f"{(seg_labeled / len(mon) * 100):.0f}%" if mon else "—"
        det_str = str(detect_labeled) if insp else "—"
        seg_str = str(seg_labeled) if mon else "—"
        print(f"{sid:<44} {first.get('machine_id',''):<10} {first.get('mold_id',''):<12} "
              f"{total_raw:>6} {det_str:>8} {seg_str:>6} {coverage_pct:>10}")
        matching.append(sid)

    print("─" * len(header))
    print(f"{len(matching)} session(s) matched.")
    return matching
