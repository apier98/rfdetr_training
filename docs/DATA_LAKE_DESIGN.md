# ARIA Data Lake — Design & Implementation Plan

> **Status**: Design approved. Implementation pending.  
> **Scope**: `ARIA_MoldVision` — `moldvision lake` CLI surface, local folder layout, and
> migration path toward low-cost remote storage.  
> Keep this document in sync with `ARIA_System_Integration.md` when integration contracts change.

---

## 1. Problem Statement

MoldPilot records qualification sessions (MP4 videos + manifest JSON). MoldTrace extracts
raw JPEG frames from those videos and separately produces process-parameter coupling data
(params ↔ defect occurrences) — but **MoldTrace does not produce image-level annotations
for CV model training**.

Image annotations (bounding boxes for defect detection, polygons for monitor segmentation)
must be created by the ARIA team using **Label Studio** inside MoldVision. This means:

- Raw frames flow **in** from MoldTrace → Data Lake (unlabeled)
- Annotators label a **selected subset** of frames via Label Studio with ML pre-labeling
- Labeled exports flow **back** into the Data Lake (annotated)
- Training datasets are **pulled** from the lake with distribution rules applied

**Partial labeling is the intended normal state.** A session may produce thousands of frames;
only a carefully chosen subset will ever be annotated. The data lake explicitly models this:
every image has an independent annotation status, and no command ever assumes full coverage.

This document designs the **ARIA Data Lake**: a local-first, folder-based image and annotation
store with a structured labeling workflow, full lineage traceability, a pull engine for
balanced training datasets, and a clear low-cost migration path to remote object storage.

---

## 2. Three-Software Integration

The data lake is the integration point for all three ARIA systems. It is managed entirely
from within `ARIA_MoldVision`.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  ARIA_MoldPilot  (shop floor)                                                ║
║  ① Operator runs Qualification Mode                                          ║
║  ② Records: inspection video + HMI video + session_manifest.json            ║
║  ③ [Future] Auto-uploads session bundle to S3 / shared path                 ║
╚══════════════════════════╤═══════════════════════════════════════════════════╝
                           │  MP4 videos + session_manifest.json
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  ARIA_MoldTrace  (pipeline)                                                  ║
║  ① extract_frames  → inspection JPEGs + monitor JPEGs (5 fps, activity-gated)║
║  ② extract_monitor → warped HMI frames (quality-filtered)                   ║
║  ③ …coupling stage → process params ↔ defect occurrences JSONL              ║
║     (future Startup Assistant input — separate from CV training)             ║
╚══════════════════════════╤═══════════════════════════════════════════════════╝
                           │  raw JPEGs only  (+ session.json with MoldPilot fields)
                           │
                           │  moldvision lake session import
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  ARIA DATA LAKE  (managed by ARIA_MoldVision)                                ║
║                                                                              ║
║  sessions/<id>/inspection_frames/  ← unlabeled on arrival                   ║
║  sessions/<id>/monitor_frames/     ← unlabeled on arrival                   ║
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │  LABELING LOOP                                                        │   ║
║  │  lake label-batch create  — select subset of frames by rules         │   ║
║  │  Label Studio + ML pre-labeling + human review                       │   ║
║  │  lake label-batch commit  — write COCO back; update lineage index    │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  lake pull  →  datasets/<UUID>/  (balanced, distribution rules applied)     ║
╚══════════════════════════╤═══════════════════════════════════════════════════╝
                           │
                           ▼
         moldvision train / export / bundle
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
         detect .mpk              seg .mpk
    Component_Base + defects    HMI_Screen quad
    (IDs 0–4, 5-class model)    (1-class seg model)
               │                       │
     ┌─────────┴─────────┐            │
     ▼                   ▼            ▼
MoldPilot           MoldTrace    MoldTrace only
monitoring mode     components   monitor_segmenter
(severity metrics   role         role
 need Component_Base
 + defect classes)
```

**What MoldTrace coupling data is NOT used for here**: the JSONL records linking process
parameters to defect occurrences are training material for the future **Startup Assistant
suggestion model** (tabular/tree/MLP). That is a separate data pipeline and model type —
not CV image annotation.

---

## 3. Guiding Principles

1. **Local-first, zero cost to start.** Everything runs on a dev workstation or NAS.
2. **Session-centric.** Images belong to sessions. Session metadata (machine, mold, operator,
   markers) is the primary filtering and audit axis.
3. **Partial annotation is the norm, not an exception.** Thousands of raw frames may exist
   per session; only a selected subset is ever annotated. The design never requires or assumes
   full session coverage.
4. **Frame selection is explicit and reproducible.** When creating a labeling batch, the
   user specifies the selection strategy (random, temporal spread, frame-gap rules). The
   exact selection is recorded so it can be reviewed or reproduced.
5. **Complete lineage traceability.** Every labeled image carries a chain:
   `session → frame → label_batch → annotation_commit → dataset_uuid → model_bundle`.
   No link in this chain is implicit or inferred.
6. **Pull-raw / push-labeled workflow.** The labeling loop has two explicit tracked steps:
   batch-create (raw → Label Studio) and batch-commit (COCO export → lake).
7. **Distribution rules are first-class.** `lake pull` enforces per-session caps and class
   balancing before handing off to the existing training pipeline.
8. **Compatible with the existing pipeline.** `lake pull` writes into `datasets/<UUID>/`.
   Everything from `moldvision dataset validate` onward is unchanged.
9. **Cloud-agnostic storage abstraction.** Local → R2/B2/S3 requires changing one config
   key; all commands stay identical.

---

## 4. Available Session Metadata

Qual-session metadata originates in MoldPilot and is preserved verbatim in the lake.

| Field | Type | Notes |
|-------|------|-------|
| `session_id` | string | e.g. `qual_20260324T093000Z_a1b2c3d4` |
| `machine_id` | string | e.g. `machine_01` |
| `mold_id` | string | e.g. `mold_a12` |
| `part_id` | string | e.g. `part_cap_32` |
| `started_at` | ISO-8601 UTC | Recording start |
| `ended_at` | ISO-8601 UTC | Recording end |
| `status` | string | `completed`, `aborted`, … |
| `operator_name` | string | |
| `batch_number` | string | e.g. `LOT-2026-0324-A` |
| `operator_notes` | string | freeform |
| `markers` | list[string] | `["setup changed", "visible flash on shot 12"]` |
| `video_chunks` | list[string] | MP4 filenames recorded |

---

## 5. Folder Layout

```
aria_data_lake/                              ← root; ARIA_DATA_LAKE env var
│                                              default: %LOCALAPPDATA%\ARIA\DataLake
│
├── data_lake_config.json                    ← created by `lake init`
│
├── sessions/                                ← one folder per qual session (write-once)
│   └── <session_id>/
│       ├── session_meta.json                ← full MoldPilot manifest
│       ├── inspection_frames/               ← component-view JPEGs (from MoldTrace)
│       │   └── frame_<N>.jpg
│       ├── monitor_frames/                  ← HMI-view JPEGs (from MoldTrace)
│       │   └── frame_<N>.jpg
│       └── annotations/
│           ├── detect/
│           │   └── _annotations.coco.json   ← COCO bbox labels; absent until committed
│           └── seg/
│               └── _annotations.coco.json   ← COCO polygon labels; absent until committed
│
├── image_index.jsonl                        ← global flat catalogue (one record / image)
│                                              — the traceability backbone
│
├── label_batches/                           ← staging area for the labeling loop
│   └── <batch_id>/
│       ├── batch_meta.json                  ← task, session sources, selection params,
│       │                                      frame list, status, created_at
│       ├── images/                          ← copies of selected frames
│       └── export/                          ← COCO export from Label Studio lands here
│
├── pools/
│   ├── hard_negatives/
│   │   └── manifest.jsonl
│   └── backgrounds/
│       └── manifest.jsonl
│
├── datasets/                                ← standard MoldVision training datasets
│   └── <UUID>/
│       ├── METADATA.json                    ← includes full lake pull provenance
│       ├── raw/
│       ├── coco/ train/ valid/ test/
│       ├── models/
│       ├── exports/
│       └── deploy/
│
└── models/                                  ← trained-model registry
    ├── defect_detection/
    │   ├── registry.json
    │   └── <bundle_id>/
    │       ├── manifest.json
    │       ├── model.onnx
    │       ├── model_fp16.onnx
    │       ├── preprocess.json
    │       └── postprocess.json
    └── monitor_segmentation/
        ├── registry.json
        └── <bundle_id>/
```

---

## 6. Traceability — Full Lineage Chain

Every artifact in the system carries an unbroken chain of provenance. No link is inferred.

```
MoldPilot session_manifest.json
  └─ session_id, machine_id, mold_id, part_id, operator, markers, timestamps
       │
       ▼  (lake session import)
  image_index.jsonl record
  ├── rel_path, session_id, machine_id, mold_id, frame_idx, frame_type
  ├── detect_status, seg_status
  └── label_batch_id (set when labeled; null when unlabeled)
       │
       ▼  (lake label-batch create)
  label_batches/<batch_id>/batch_meta.json
  ├── batch_id, task, created_at, status
  ├── selection: { mode, seed, n, min_frame_gap, sessions, filters }
  └── frames: [ list of rel_paths selected ]
       │
       ▼  (Label Studio annotation + lake label-batch commit)
  sessions/<id>/annotations/<task>/_annotations.coco.json
  ├── Standard COCO format
  └── images[*].extra.label_batch_id   ← non-standard field; ignored by RF-DETR
       │
       ▼  (lake pull)
  datasets/<UUID>/METADATA.json
  ├── uuid, name, created_at, class_names
  └── lake_pull_provenance:
      ├── task, seed, train_ratio
      ├── sessions: [ {session_id, frames_selected, frames_total_labeled} ]
      ├── filters: { machine_id, mold_id, from, to, markers }
      ├── distribution: { max_per_session, balance_classes, min_per_class }
      └── pools: { hard_negatives: N, backgrounds: N }
       │
       ▼  (moldvision train + bundle)
  models/<task>/registry.json bundle entry
  ├── bundle_id, version, channel, created_at
  └── dataset_uuid  ← points back to METADATA.json
```

This chain means you can always answer:
- "Which session does this image come from?" → `image_index.jsonl`
- "Who labeled it and in which batch?" → `label_batch_id` in index + `batch_meta.json`
- "Which frames were in that batch and how were they selected?" → `batch_meta.json`
- "Which dataset was used to train this model?" → `registry.json → dataset_uuid`
- "Which sessions and rules built that dataset?" → `METADATA.json → lake_pull_provenance`
- "What model is deployed in production right now?" → `registry.json → active.stable`

### 6.1 `image_index.jsonl` — record schema

```json
{
  "rel_path":       "sessions/qual_20260324T093000Z_a1b2c3d4/inspection_frames/frame_000100.jpg",
  "session_id":     "qual_20260324T093000Z_a1b2c3d4",
  "machine_id":     "machine_01",
  "mold_id":        "mold_a12",
  "part_id":        "part_cap_32",
  "operator_name":  "Andrea Rossi",
  "batch_number":   "LOT-2026-0324-A",
  "started_at":     "2026-03-24T09:30:00Z",
  "markers":        ["setup changed"],
  "frame_type":     "inspection",
  "frame_idx":      100,
  "detect_status":  "labeled",
  "detect_batch_id":"batch-2026-q2-detect-a3f1c2",
  "seg_status":     "n/a",
  "seg_batch_id":   null
}
```

`detect_status` / `seg_status` values:

| Value | Meaning |
|-------|---------|
| `unlabeled` | No annotation committed yet |
| `labeled` | COCO annotation committed via a label-batch commit |
| `hard_negative` | Confirmed no-defect (in `pools/hard_negatives/`) |
| `background` | No component visible (in `pools/backgrounds/`) |
| `n/a` | Frame type not applicable to this task |

`detect_batch_id` / `seg_batch_id`: the `batch_id` of the label-batch that produced the
annotation. Null when `unlabeled`.

### 6.2 `batch_meta.json` — schema

```json
{
  "batch_id":     "batch-2026-q2-detect-a3f1c2",
  "task":         "detect",
  "status":       "committed",
  "created_at":   "2026-04-07T10:00:00Z",
  "committed_at": "2026-04-08T14:30:00Z",
  "selection": {
    "mode":          "temporal",
    "n":             200,
    "min_frame_gap": 10,
    "seed":          42,
    "sessions":      ["qual_20260324T093000Z_a1b2c3d4"],
    "filters":       {"mold_id": "mold_a12"}
  },
  "frames": [
    "sessions/qual_.../inspection_frames/frame_000000.jpg",
    "sessions/qual_.../inspection_frames/frame_000010.jpg"
  ],
  "commit_summary": {
    "images_committed": 187,
    "images_skipped_not_in_export": 13,
    "annotations_per_class": {
      "Component_Base": 187, "Weld_Line": 42, "Sink_Mark": 28,
      "Flash": 11, "Burn_Mark": 6
    }
  }
}
```

### 6.3 `METADATA.json` — extended schema (lake pull provenance)

```json
{
  "uuid":        "3f9a1c2b-...",
  "name":        "mold-a12-detect-2026-q2",
  "created_at":  "2026-04-10T09:00:00Z",
  "class_names": ["Component_Base", "Weld_Line", "Sink_Mark", "Flash", "Burn_Mark"],
  "notes":       "",
  "lake_pull_provenance": {
    "task":        "detect",
    "seed":        42,
    "train_ratio": 0.85,
    "sessions": [
      {"session_id": "qual_20260324...", "frames_selected": 200, "frames_total_labeled": 320},
      {"session_id": "qual_20260401...", "frames_selected": 540, "frames_total_labeled": 540}
    ],
    "filters":      {"mold_id": "mold_a12"},
    "distribution": {"max_per_session": 200, "balance_classes": true, "min_per_class": 40},
    "pools":        {"hard_negatives": 40, "backgrounds": 0}
  }
}
```

---

## 7. Two Model Tasks

| | `detect` | `seg` |
|---|---|---|
| **Purpose** | Locate the component AND detect its defects — both required for severity metrics | Monitor screen quad localization |
| **Source frames** | `inspection_frames/` | `monitor_frames/` |
| **Annotation type** | Bounding boxes | Polygons |
| **Tool** | Label Studio (RectangleLabels) | Label Studio (PolygonLabels) |
| **Lake annotation path** | `annotations/detect/` | `annotations/seg/` |
| **Class schema** | **Fixed** — see §7.1 | `HMI_Screen` (1 class) |
| **Consumers** | MoldPilot (monitoring) **and** MoldTrace (`components` role) | MoldTrace only (`monitor_segmenter` role) |

### 7.1 Defect Detection — Fixed Class IDs

The detect model is a **single 5-class model** deployed to both MoldPilot and MoldTrace.
`Component_Base` and defect classes must always be trained together: MoldPilot anchors
severity metrics against the `Component_Base` box; MoldTrace uses both in its pipeline.

| ID | Class name | Role |
|----|-----------|------|
| 0 | `Component_Base` | Anchor box — required for IoU tracking and severity computation |
| 1 | `Weld_Line` | Defect class |
| 2 | `Sink_Mark` | Defect class |
| 3 | `Flash` | Defect class |
| 4 | `Burn_Mark` | Defect class |

> ⚠️ Changing class IDs requires a coordinated update across MoldPilot, MoldTrace, and
> all deployed bundles. Treat these as immutable.
>
> **Annotation rule**: every inspection frame that shows defects must also have a
> `Component_Base` box. Annotations with defects but no `Component_Base` are invalid.

---

## 8. `moldvision lake` CLI Surface

### 8.1 `lake init`

```
moldvision lake init [--root PATH]
```

Creates the folder skeleton and writes `data_lake_config.json`.

---

### 8.2 `lake import` — external data

Import images from outside the MoldPilot workflow (historical data, supplier datasets,
public datasets, re-exports from other tools). The imported data becomes a **fully
first-class session** in the lake: it participates in `lake pull`, `--max-per-session`
distribution rules, `lake session list`, and the traceability chain identically to data
that originated from MoldPilot.

```
moldvision lake import
  --images-dir  <dir>                 # JPEG/PNG images to import (required)
  --task        detect|seg            # 'detect' → inspection frames | 'seg' → monitor frames

  # Optional pre-existing annotations
  [--coco-json  <file>]               # COCO annotation file.
                                      # Images with annotations → 'labeled'
                                      # Images without annotations → 'unlabeled'
                                      # (partial annotation is explicitly supported)

  # Session identity and metadata
  [--session-id <id>]                 # custom ID; default: 'external_<ts>_<uuid>'
  [--name       "supplier batch A"]   # human-readable label
  [--machine-id X]
  [--mold-id    Y]
  [--part-id    Z]
  [--notes      "from supplier XYZ, 2026-Q1"]

  [--overwrite]
  [--lake-root  PATH]
```

Actions:
1. Generates a session ID with prefix `external_` (clearly distinguishable from MoldPilot sessions).
2. Copies images into `sessions/<id>/inspection_frames/` or `monitor_frames/`.
3. Writes a synthetic `session_meta.json` with the supplied metadata.
4. If `--coco-json` is given: writes the annotations into `sessions/<id>/annotations/<task>/` and
   marks the annotated images as `labeled` immediately. Unannotated images remain `unlabeled`.
5. Appends records to `image_index.jsonl`.

> **Design rationale**: external data should not have a different code path in `lake pull`
> — it just looks like a session that happened to arrive pre-labeled.  
> The `external_` prefix in the session ID is a convention only; it has no special behaviour.

---

### 8.3 `lake session import`

Registers a new qual session and its raw frames. Called after MoldTrace finishes
`extract_frames` for a session. No annotation happens here — frames are raw material.

```
moldvision lake session import
  --session-meta  <path/to/session.json>    # MoldPilot manifest (via MoldTrace meta/)
  --inspection-frames <dir>                 # extracted component-view JPEGs
  [--monitor-frames   <dir>]                # extracted HMI-view JPEGs
  [--overwrite]
```

Actions:
1. Copies frames into `sessions/<id>/inspection_frames/` and `monitor_frames/`.
2. Writes `session_meta.json` from MoldPilot manifest fields.
3. Appends `unlabeled` records to `image_index.jsonl` for every frame.

---

### 8.4 `lake session list`

```
moldvision lake session list
  [--machine-id X] [--mold-id Y] [--part-id Z]
  [--from 2026-01-01] [--to 2026-06-01]
  [--task detect|seg]
  [--label-status labeled|unlabeled|any]
  [--marker "visible flash"]
  [--min-frames N]
```

Output:

```
session_id                            machine  mold      raw     detect  seg  coverage
qual_20260324T093000Z_a1b2c3d4        mach_01  mold_a12  1 240     320   120    26%
qual_20260401T143000Z_f8e9d3c1        mach_01  mold_a12    540     540    —    100%
```

`coverage` = labeled / raw, for the selected `--task`.

---

### 8.5 Labeling Loop: `lake label-batch create` / `commit`

Partial labeling is expected. A session may have 1 000 frames but only 80 will be sent to
Label Studio. The batch commands make that selection explicit and auditable.

#### `lake label-batch create`

```
moldvision lake label-batch create
  --task detect|seg

  # Session / image filter
  --sessions s1,s2,...
  --all
  --machine-id X  --mold-id Y
  --marker "setup changed"
  --only-unlabeled              # default: True — skip already-labeled frames

  # Frame selection rules (how to choose which frames within sessions)
  --n 200                       # total frames to select across all matched sessions
  --sample-mode random          # random uniform (default)
  --sample-mode temporal        # evenly spaced across each session's timeline
  --min-frame-gap N             # minimum frame_idx gap between selected frames
                                # (avoids near-duplicate consecutive frames)
  --skip-first N                # ignore first N frames of each session
                                # (skips camera startup / setup noise)
  --skip-last  N                # ignore last N frames (teardown)
  --seed 42                     # makes selection fully reproducible

  # Output
  --batch-name "batch-2026-q2-detect"
  [--batch-root PATH]
```

The `--min-frame-gap` and `--sample-mode temporal` flags are the primary tools for avoiding
the most common labeling pitfall: selecting 200 nearly-identical consecutive frames from one
area of the session, which contributes little diversity to the model.

Actions:
1. Queries `image_index.jsonl` for `unlabeled` frames matching filters.
2. Groups by session; stratifies to distribute `--n` proportionally across sessions.
3. Applies frame selection rules within each session.
4. Copies selected frames into `label_batches/<batch_id>/images/`.
5. Writes `batch_meta.json` with full selection parameters and frame list.
6. Prints Label Studio instructions.

```
Batch created: label_batches/batch-2026-q2-detect-a3f1c2/
Images: 200  (from 3 sessions: qual_20260324=80, qual_20260401=80, qual_20260405=40)
Sample mode: temporal  |  min-frame-gap: 10  |  seed: 42

Next steps:
1. Start the ML backend (pre-labeling):
   moldvision label-studio-backend --task detect \
     --bundle models/defect_detection/mold-defect-v1.0.0/

2. In Label Studio: create a project → add ML backend → import images from:
   label_batches/batch-2026-q2-detect-a3f1c2/images/

3. Annotate, then export COCO JSON to:
   label_batches/batch-2026-q2-detect-a3f1c2/export/

4. Commit:
   moldvision lake label-batch commit --batch batch-2026-q2-detect-a3f1c2
```

#### `lake label-batch commit`

```
moldvision lake label-batch commit
  --batch <batch_id>
  [--coco-json PATH]          # explicit path; default: batch/export/*.json
  [--dry-run]
```

Actions:
1. Reads the COCO JSON from `batch/export/`.
2. Merges annotations into `sessions/<session_id>/annotations/<task>/_annotations.coco.json`.
3. Updates `image_index.jsonl`: sets `detect_status = "labeled"` and `detect_batch_id`
   for every committed image.
4. Updates `batch_meta.json`: status → `committed`, records `commit_summary`.
5. Prints summary: images committed per session, annotations per class.

> **Partial commits are safe and expected.** If Label Studio only annotated 150 of 200
> batch images, the remaining 50 stay `unlabeled` in the index and can be included
> in the next batch. The system never assumes a batch was fully annotated.

---

### 8.6 `lake pull` — build training dataset

Assembles a training-ready `datasets/<UUID>/` from labeled images across sessions.
The full set of parameters used is stored in `METADATA.json` for reproducibility.

```
moldvision lake pull
  --task detect|seg

  # Session selection
  --sessions s1,s2,s3
  --all
  --machine-id X  --mold-id Y  --part-id Z
  --from 2026-01-01  --to 2026-06-01
  --marker "setup changed"

  # Annotation filter
  --label-status labeled            # default
  --include-hard-negatives
  --include-backgrounds

  # Distribution rules
  --max-per-session N               # cap any single session's contribution
  --min-per-session N               # skip sessions below this count
  --balance-classes                 # undersample to equalise per-class counts
  --min-per-class N                 # abort if any class has < N annotations
  --train-ratio 0.8
  --seed 42

  # Output
  --dataset-uuid UUID
  --dataset-name "my-dataset"
  --dry-run
```

#### Pull algorithm

```
1. Read image_index.jsonl — filter by task, detect_status=labeled, and all --filter flags
2. Group images by session_id
3. Drop sessions below --min-per-session
4. Apply --max-per-session: random-sample within each session (seeded)
5. Collect COCO annotations from sessions/<id>/annotations/<task>/_annotations.coco.json
6. Merge COCO dicts across sessions (re-number IDs to avoid collisions)
7. Append hard-negatives / backgrounds pools if requested
8. If --balance-classes: undersample over-represented classes to 2× rarest count
9. Shuffle and split train/valid (--train-ratio, --seed)
10. Call create_dataset() + copy images into datasets/<UUID>/raw/
11. Write merged COCO JSONs to datasets/<UUID>/coco/train/ and valid/
12. Write lake_pull_provenance into datasets/<UUID>/METADATA.json
13. Print distribution report
```

#### Dry-run output

```
lake pull DRY RUN — task=detect
──────────────────────────────────────────────────────────────────
Sessions (after filters):
  qual_20260324...  mach_01  mold_a12  320 labeled → capped at 200
  qual_20260401...  mach_01  mold_a12  540 labeled → all 540 included
  qual_20260405...  mach_01  mold_a12   48 labeled → skipped (< --min-per-session 50)

Class distribution (before balance):
  Component_Base  1 520  ██████████████████████
  Weld_Line         298  ████
  Sink_Mark         187  ███
  Flash              72  █
  Burn_Mark          38  ▌
  hard_negatives     40  ▌

After --balance-classes (2× rarest = Burn_Mark × 2 = 76):
  Component_Base   76
  Weld_Line        76
  Sink_Mark        76
  Flash            72
  Burn_Mark        38

Train: 290 images / 338 annotations
Valid:  73 images /  84 annotations

Provenance will be saved to METADATA.json → lake_pull_provenance
──────────────────────────────────────────────────────────────────
Run without --dry-run to create dataset.
```

---

### 8.7 `lake index`

```
moldvision lake index --rebuild          # full scan → rewrite image_index.jsonl
moldvision lake index --stats [--task detect|seg]
```

`--rebuild` is the recovery command: run after manually copying frames, after an out-of-band
annotation edit, or to repair a corrupted index.

---

### 8.8 `lake models install / list / promote`

```
moldvision lake models install <bundle.mpk> --task detect|seg
moldvision lake models list    [--task detect|seg]
moldvision lake models promote <bundle_id> --channel stable|dev
```

`install` extracts `.mpk` into `models/<task>/<bundle_id>/` and appends to `registry.json`.
`promote` updates `active.<channel>` — selects which bundle the Label Studio ML backend
defaults to for pre-labeling, closing the active-learning loop.

---

## 9. Hard Negatives and Backgrounds

### 9.1 Hard Negatives

Images where the model predicted a defect but a human confirmed there was none.
Most impactful data for reducing false-positive rates.

```
moldvision lake pools add-hard-negative \
  --image sessions/<id>/inspection_frames/frame_000842.jpg \
  --reason "model_false_positive"
```

### 9.2 Backgrounds

Images with no component visible (empty conveyor, setup shots).

```
moldvision lake pools add-background \
  --images sessions/<id>/inspection_frames/frame_00100*.jpg
```

Both pools are ingested as empty-annotation COCO images when `lake pull --include-*` is used.
Their `detect_status` in `image_index.jsonl` is set to `hard_negative` / `background`
(not `labeled`) so they are independently controlled.

---

## 10. Complete End-to-End Workflow

```
① MoldPilot  — operator runs Qualification Mode
   → produces: *.mp4 chunks + session_manifest.json

② MoldTrace  — extract frames
   python -m moldtrace run --session <uuid> --extract-frames --extract-monitor

③ MoldVision — import into the data lake
   moldvision lake session import \
     --session-meta  <moldtrace>/meta/session.json \
     --inspection-frames <moldtrace>/inputs/inspection_video/frames/vid_01/ \
     --monitor-frames    <moldtrace>/inputs/process_video/frames/vid_01/
   → image_index.jsonl updated; all frames unlabeled

④ MoldVision — review coverage
   moldvision lake session list --task detect --label-status unlabeled

⑤ MoldVision — create a labeling batch (select subset of frames)
   moldvision lake label-batch create \
     --task detect --mold-id mold_a12 \
     --n 300 --sample-mode temporal --min-frame-gap 8 \
     --skip-first 30 --seed 42 \
     --batch-name "mold-a12-detect-batch-01"
   → 300 frames selected, copied to label_batches/..., batch_meta.json written

⑥ MoldVision — start Label Studio ML backend (pre-labeling)
   moldvision label-studio-backend --task detect \
     --bundle models/defect_detection/mold-defect-v1.0.0/

⑦ Annotation — label in Label Studio
   → import label_batches/mold-a12-detect-batch-01/images/
   → ML backend pre-labels frames; annotator corrects
   → export COCO JSON → label_batches/mold-a12-detect-batch-01/export/

⑧ MoldVision — commit labels back to the lake
   moldvision lake label-batch commit --batch mold-a12-detect-batch-01
   → annotations written to sessions/<id>/annotations/detect/_annotations.coco.json
   → image_index.jsonl updated: detect_status=labeled, detect_batch_id=batch-...

⑨ MoldVision — preview training distribution
   moldvision lake pull --task detect --all \
     --max-per-session 250 --balance-classes --dry-run

⑩ MoldVision — pull training dataset
   moldvision lake pull \
     --task detect --all \
     --include-hard-negatives \
     --max-per-session 250 --balance-classes --min-per-class 40 \
     --train-ratio 0.85 --seed 42 \
     --dataset-name "mold-defect-2026-q2"
   → datasets/<UUID>/ created with METADATA.json including lake_pull_provenance

⑪ MoldVision — validate and train
   moldvision dataset validate -d <UUID> --task detect
   moldvision train -d <UUID> --task detect --epochs 60 --batch-size 4

⑫ MoldVision — export and bundle
   moldvision export -d <UUID> -w checkpoint_best_total.pth --format onnx_fp16
   moldvision bundle -d <UUID> -w checkpoint_best_total.pth \
     --model-name mold-defect --model-version 2.1.0 --mpk

⑬ MoldVision — register in model registry
   moldvision lake models install \
     datasets/<UUID>/deploy/mold-defect-v2.1.0.mpk --task detect
   moldvision lake models promote mold-defect-v2.1.0 --channel stable

⑭ Deploy to consumers
   → MoldPilot: install bundle (monitoring mode, Component_Base + defects)
   → MoldTrace: install bundle as `components` role
   → Label Studio ML backend: automatically uses the new `stable` bundle for next batch
```

---

## 11. Remote Storage — Future Migration Path

The data lake is designed so only the **storage backend changes** when moving to the cloud.
All CLI commands, annotation workflows, and training pipelines stay identical.

### 11.1 Why Move to Remote Storage?

| Trigger | Reason |
|---------|--------|
| Second annotator / remote team | Shared access to the same frames and annotations |
| MoldTrace moves to AWS | Frames extracted on EC2; round-tripping to local wastes bandwidth |
| Backup / disaster recovery | Irreplaceable labeling work needs off-machine redundancy |
| Dataset > 50 GB | Outgrows a single workstation |

### 11.2 Backend Options — Cost First

| Option | Storage /GB/mo | Egress /GB | S3 API | Notes |
|--------|---------------|------------|:---:|-------|
| **Local / NAS** | HW only | Free | — | Phase 0. Zero recurring cost. |
| **MinIO (self-hosted)** | HW only | Free | ✅ | NAS or internal server. Full S3 API. Scale without cost increase. |
| **Backblaze B2** | $0.006 | $0.01 (free to CDN) | ✅ | ~4× cheaper than S3 for storage. Good first paid option. |
| **Cloudflare R2** | $0.015 | **$0.00** | ✅ | Zero egress. Best when MoldPilot and MoldTrace pull bundles frequently. |
| **AWS S3** | $0.023 | $0.09 | ✅ | Most expensive. Justified only if team is fully committed to AWS for MoldTrace. |
| **DagsHub free tier** | 5 GB free | Free | via DVC | Prototype DVC versioning at zero cost before committing to a paid backend. |

**Recommended progression:**
1. **Now** — local filesystem or NAS share (zero cost).
2. **When a second person needs access** — MinIO on a NAS (free, S3 API, no internet needed).
3. **When MoldTrace moves to AWS** — Cloudflare R2 (zero egress prevents surprises when
   MoldPilot polls for bundle updates or MoldTrace uploads frame batches).

### 11.3 Abstraction Layer

```python
class ILakeStorage(Protocol):
    def exists(self, rel_path: str) -> bool: ...
    def read_bytes(self, rel_path: str) -> bytes: ...
    def write_bytes(self, rel_path: str, data: bytes) -> None: ...
    def list_prefix(self, prefix: str) -> list[str]: ...
    def delete(self, rel_path: str) -> None: ...
```

- `LocalLakeStorage(root: Path)` — filesystem, active now
- `S3LakeStorage(bucket, prefix, endpoint_url, client)` — AWS S3, R2, B2, MinIO

All `rel_path` values are relative, so portable between backends without rewriting.

Switching backend = one key change in `data_lake_config.json`:
```json
{
  "backend": "s3",
  "s3_bucket": "aria-data-lake",
  "s3_prefix": "v1/",
  "s3_endpoint_url": "https://<account>.r2.cloudflarestorage.com"
}
```

### 11.4 DVC as an Optional Versioning Layer

DVC can version the lake on top of any S3-compatible backend, giving git-tracked dataset
snapshots (i.e. "dataset_v3 used sessions A+B+C at annotation commit X"):

```bash
dvc init
dvc remote add -d lake s3://aria-data-lake/dvc
dvc add aria_data_lake/sessions/
dvc push
```

Nothing in the local lake design blocks this; DVC is added incrementally with no subscription.

### 11.5 Migration Checklist (Local → Cloudflare R2)

1. Create R2 bucket `aria-data-lake`.
2. Generate R2 API token (S3-compatible: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).
3. Update `data_lake_config.json`: `"backend": "s3"` + endpoint + bucket.
4. Run `moldvision lake sync --upload` to push sessions, annotations, and models.
5. Update `scripts/import_from_moldtrace.py` to write directly to R2.
6. All `lake session import`, `lake label-batch create/commit`, `lake pull` continue unchanged.

---

## 12. Implementation Phases

### Phase 1 — Lake skeleton, session import, index

New files: `moldvision/lake.py`, `moldvision/lake_storage.py` (local only at this stage)

Delivered: `lake init`, `lake session import`, `lake index --rebuild / --stats`,
`lake session list`

Acceptance test:
```powershell
moldvision lake init --root C:\dev\aria_data_lake
moldvision lake session import --session-meta session.json --inspection-frames frames/
moldvision lake index --stats
moldvision lake session list --task detect
```

### Phase 2 — Labeling loop

New files: `moldvision/lake_label.py`

Delivered: `lake label-batch create` (all sample modes), `lake label-batch commit`

Acceptance test:
```powershell
moldvision lake label-batch create --task detect --all --n 50 \
  --sample-mode temporal --min-frame-gap 5 --batch-name smoke
# drop a minimal COCO JSON into label_batches/smoke.../export/
moldvision lake label-batch commit --batch smoke...
moldvision lake index --stats   # detect_status=labeled for committed images
                                # detect_batch_id correctly set
```

### Phase 3 — Pull engine + pools

New files: `moldvision/lake_pull.py`

Delivered: `lake pull` (all flags, provenance in METADATA.json),
`lake pools add-hard-negative / add-background`

Acceptance test:
```powershell
moldvision lake pull --task detect --all \
  --max-per-session 200 --balance-classes --dry-run
moldvision lake pull --task detect --all \
  --max-per-session 200 --balance-classes --dataset-name smoke-test
# verify METADATA.json contains lake_pull_provenance
moldvision dataset validate -d <UUID> --task detect
```

### Phase 4 — Model registry

New files: `moldvision/lake_models.py`

Delivered: `lake models install / list / promote`

### Phase 5 — Storage backend + sync

Update `moldvision/lake_storage.py` to add `S3LakeStorage`.

Delivered: `lake sync --upload / --download`

### Phase 6 — MoldTrace integration script

New file (in `ARIA_MoldTrace`): `scripts/import_session_to_lake.py`

Post-pipeline hook that calls `moldvision lake session import` automatically with the
correct MoldTrace output paths. No MoldVision code changes required.

---

## 13. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARIA_DATA_LAKE` | `%LOCALAPPDATA%\ARIA\DataLake` | Data lake root |
| `ARIA_LAKE_BACKEND` | `local` | `local` or `s3`; overrides `data_lake_config.json` |
| `ARIA_LAKE_S3_BUCKET` | — | Bucket name |
| `ARIA_LAKE_S3_ENDPOINT` | — | Custom endpoint URL (R2 / B2 / MinIO) |
| `ARIA_LAKE_S3_PREFIX` | `v1/` | Key prefix inside bucket |

---

## 14. Open Questions

| # | Question | Recommendation |
|---|----------|---------------|
| 1 | `sessions/` and `datasets/` under the same `aria_data_lake/` root? | Yes — one env var, one sync command. |
| 2 | Should `label-batch create` copy images or symlink them? | Copies on Windows (symlinks require admin rights); symlinks on Linux/Mac for speed. |
| 3 | `--balance-classes`: hard undersample at pull time vs training-time class weights? | Hard undersample at pull time — deterministic and reproducible. A `--class-weights` option for training is a future addition. |
| 4 | Where do training `runs/` logs go? | `aria_data_lake/runs/<dataset_uuid>/` — keeps each dataset and its training history together. |
| 5 | Should Label Studio project setup be automated (API call) in `label-batch create`? | Future Phase — Phase 2 prints instructions; later phases can call the Label Studio API. |
| 6 | Do we need a `lake label-batch status` command to see open/committed batches? | Yes — add to Phase 2 alongside create/commit. |

> **Status**: Design approved. Implementation pending.  
> **Scope**: `ARIA_MoldVision` — `moldvision lake` CLI surface, local folder layout, and
> migration path toward low-cost remote storage.  
> Keep this document in sync with `ARIA_System_Integration.md` when integration contracts change.

---

## 1. Problem Statement

MoldPilot records qualification sessions (MP4 videos + manifest JSON). MoldTrace extracts
raw JPEG frames from those videos and separately produces process-parameter coupling data
(params ↔ defect occurrences) — but **MoldTrace does not produce image-level annotations
for CV model training**.

Image annotations (bounding boxes for defect detection, polygons for monitor segmentation)
must be created by the ARIA team using **Label Studio** inside MoldVision. This means:

- Raw frames flow **in** from MoldTrace → Data Lake (unlabeled)
- Annotators label frames in Label Studio (with pre-labeling from the ML backend)
- Labeled exports flow **back** into the Data Lake (annotated)
- Training datasets are **pulled** from the lake with distribution rules applied

Not every session will be labeled — sessions are a source of raw material and annotation
is a deliberate, selective act. The data lake must track per-image annotation status so
partial labeling is the normal state.

This document designs the **ARIA Data Lake**: a local-first, folder-based image and annotation
store with a labeling workflow, a pull engine for balanced training datasets, and a clear
low-cost migration path to remote object storage.

---

## 2. Data Flow Clarification

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  MoldPilot                                                                   │
│  Records qual session → MP4 videos + session_manifest.json                  │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ raw videos + manifest
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  MoldTrace                                                                   │
│  ① extract_frames  → inspection JPEGs + monitor JPEGs                       │
│  ② …pipeline stages…                                                        │
│  ③ coupling stage  → process params ↔ defect occurrences (JSONL)            │
│     (used for Startup Assistant suggestion logic — future, not CV training)  │
└──────────┬────────────────────────────────┬─────────────────────────────────┘
           │ raw frames only                │ coupling JSONL (future)
           ▼                                ▼ (separate pipeline, not in scope here)
┌──────────────────────────────────────────────────────────────────────────────┐
│  ARIA Data Lake  (MoldVision manages this)                                   │
│                                                                              │
│  sessions/<id>/inspection_frames/  ← unlabeled on arrival                   │
│  sessions/<id>/monitor_frames/     ← unlabeled on arrival                   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  LABELING LOOP (in MoldVision)                                         │  │
│  │                                                                        │  │
│  │  lake label-batch create  →  Label Studio project (raw images)        │  │
│  │         Label Studio + ML pre-labeling + human review                 │  │
│  │  lake label-batch commit  ←  COCO export from Label Studio            │  │
│  │                               ↓                                       │  │
│  │  sessions/<id>/annotations/detect/_annotations.coco.json              │  │
│  │  sessions/<id>/annotations/seg/_annotations.coco.json                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  lake pull  →  datasets/<UUID>/  (balanced, ready for training)              │
└──────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
              moldvision train / export / bundle
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             detect .mpk              seg .mpk
        Component_Base + defects    HMI_Screen quad
        (IDs 0–4, 5-class model)    (1-class seg model)
                    │                       │
          ┌─────────┴─────────┐            │
          ▼                   ▼            ▼
    MoldPilot           MoldTrace    MoldTrace only
    monitoring mode     components   monitor_segmenter
    (severity metrics   role         role
     require both
     Component_Base
     and defect classes)
```

**What MoldTrace coupling data is NOT used for here**: the JSONL records linking process
parameters to defect occurrences are training material for the future **Startup Assistant
suggestion model** (tabular/tree/MLP model in MoldPilot). That is a separate data pipeline
and a separate model type — not CV image annotation.

---

## 3. Guiding Principles

1. **Local-first, zero cost to start.** Everything runs on a dev workstation or NAS.
2. **Session-centric.** Images belong to sessions. Session metadata is the primary axis
   for filtering and audit trails.
3. **Partial annotation is the norm.** A session may have 1 000 raw frames but only
   80 labeled ones. The index tracks this at per-image granularity.
4. **Pull-raw / push-labeled workflow.** The labeling loop has two explicit steps:
   pull a batch of unlabeled frames into a Label Studio project, then commit the COCO
   export back. This keeps annotation and storage decoupled.
5. **Distribution rules are first-class.** `lake pull` enforces per-session caps and
   class balancing before handing off to the existing training pipeline.
6. **Compatible with the existing pipeline.** `lake pull` writes into `datasets/<UUID>/`.
   Everything from `moldvision dataset validate` onward is unchanged.
7. **Cloud-agnostic storage abstraction.** Local → R2/B2/S3 requires changing one config
   key; all commands stay identical.

---

## 4. Available Session Metadata

Qual-session metadata originates in MoldPilot and is preserved verbatim in the lake.

| Field | Type | Notes |
|-------|------|-------|
| `session_id` | string | e.g. `qual_20260324T093000Z_a1b2c3d4` |
| `machine_id` | string | e.g. `machine_01` |
| `mold_id` | string | e.g. `mold_a12` |
| `part_id` | string | e.g. `part_cap_32` |
| `started_at` | ISO-8601 UTC | Recording start |
| `ended_at` | ISO-8601 UTC | Recording end |
| `status` | string | `completed`, `aborted`, … |
| `operator_name` | string | |
| `batch_number` | string | e.g. `LOT-2026-0324-A` |
| `operator_notes` | string | freeform |
| `markers` | list[string] | `["setup changed", "visible flash on shot 12"]` |
| `video_chunks` | list[string] | MP4 filenames recorded |

These are written into `sessions/<session_id>/session_meta.json` and indexed in
`image_index.jsonl` for per-image filtering.

---

## 5. Folder Layout

```
aria_data_lake/                              ← root; ARIA_DATA_LAKE env var
│                                              default: %LOCALAPPDATA%\ARIA\DataLake
│
├── data_lake_config.json                    ← created by `lake init`
│
├── sessions/                                ← one folder per qual session (write-once)
│   └── <session_id>/
│       ├── session_meta.json                ← full MoldPilot manifest
│       ├── inspection_frames/               ← component-view JPEGs (from MoldTrace)
│       │   └── frame_<N>.jpg
│       ├── monitor_frames/                  ← HMI-view JPEGs (from MoldTrace)
│       │   └── frame_<N>.jpg
│       └── annotations/
│           ├── detect/
│           │   └── _annotations.coco.json   ← COCO bbox labels; absent until committed
│           └── seg/
│               └── _annotations.coco.json   ← COCO polygon labels; absent until committed
│
├── image_index.jsonl                        ← global flat catalogue (one record / image)
│
├── label_batches/                           ← staging area for the labeling loop
│   └── <batch_id>/
│       ├── batch_meta.json                  ← task, session IDs, created_at, status
│       ├── images/                          ← symlinks or copies of frames to label
│       └── export/                          ← COCO export from Label Studio lands here
│
├── pools/
│   ├── hard_negatives/
│   │   └── manifest.jsonl                   ← images confirmed as false-positive
│   └── backgrounds/
│       └── manifest.jsonl                   ← images with no component visible
│
├── datasets/                                ← standard MoldVision training datasets
│   └── <UUID>/
│       ├── METADATA.json
│       ├── raw/
│       ├── coco/ train/ valid/ test/
│       ├── models/
│       ├── exports/
│       └── deploy/
│
└── models/                                  ← trained-model registry
    ├── defect_detection/
    │   ├── registry.json
    │   └── <bundle_id>/                     ← extracted .mpk
    │       ├── manifest.json
    │       ├── model.onnx
    │       ├── model_fp16.onnx
    │       ├── preprocess.json
    │       └── postprocess.json
    └── monitor_segmentation/
        ├── registry.json
        └── <bundle_id>/
```

### 5.1 `image_index.jsonl` — record schema

One JSON object per line. This is the queryable backbone of `lake pull` and `lake session list`.

```json
{
  "rel_path":     "sessions/qual_20260324T093000Z_a1b2c3d4/inspection_frames/frame_000100.jpg",
  "session_id":   "qual_20260324T093000Z_a1b2c3d4",
  "machine_id":   "machine_01",
  "mold_id":      "mold_a12",
  "part_id":      "part_cap_32",
  "operator_name":"Andrea Rossi",
  "batch_number": "LOT-2026-0324-A",
  "started_at":   "2026-03-24T09:30:00Z",
  "markers":      ["setup changed"],
  "frame_type":   "inspection",
  "frame_idx":    100,
  "detect_status": "unlabeled",
  "seg_status":    "n/a"
}
```

`detect_status` / `seg_status` values:

| Value | Meaning |
|-------|---------|
| `unlabeled` | No annotation committed yet |
| `labeled` | COCO annotation committed and present |
| `hard_negative` | Marked as confirmed no-defect (in `pools/hard_negatives/`) |
| `background` | No component visible (in `pools/backgrounds/`) |
| `n/a` | Frame type not applicable to this task |

Each task has its own status column because the same inspection frame might be labeled for
`detect` but not for `seg`, and vice versa for monitor frames.

### 5.2 `models/*/registry.json` — schema

```json
{
  "task": "defect_detection",
  "bundles": [
    {
      "bundle_id":    "mold-defect-v2.0.0",
      "version":      "2.0.0",
      "channel":      "stable",
      "dataset_uuid": "3f9a1c2b-...",
      "created_at":   "2026-04-07T10:00:00Z",
      "path":         "models/defect_detection/mold-defect-v2.0.0/"
    }
  ],
  "active": {
    "stable": "mold-defect-v2.0.0",
    "dev":    "mold-defect-v2.1.0-dev"
  }
}
```

---

## 6. Two Model Tasks

| | `detect` | `seg` |
|---|---|---|
| **Purpose** | Locate the component AND detect its defects in a single pass — both are required to compute severity metrics | Monitor screen quad localization |
| **Source frames** | `inspection_frames/` | `monitor_frames/` |
| **Annotation type** | Bounding boxes | Polygons |
| **Tool** | Label Studio (RectangleLabels) | Label Studio (PolygonLabels) |
| **Lake annotation path** | `annotations/detect/` | `annotations/seg/` |
| **Class schema** | **Fixed** — see §6.1 | `HMI_Screen` (1 class) |
| **Consumers** | MoldPilot (monitoring) **and** MoldTrace (`components` role) | MoldTrace only (`monitor_segmenter` role) |

### 6.1 Defect Detection — Fixed Class IDs

The detect model is a **single 5-class model** used by both MoldPilot and MoldTrace.
`Component_Base` and the defect classes must always be trained together: MoldPilot's severity
metrics are computed by anchoring defect boxes against the `Component_Base` bounding box, so
a model that detects only defects is unusable. A model that detects only components is equally
incomplete.

Hard contract shared with MoldPilot (`OnnxInferenceService`) and MoldTrace (`components` role).

| ID | Class name | Role |
|----|-----------|------|
| 0 | `Component_Base` | Anchor: locates the part; required for IoU tracking and severity computation |
| 1 | `Weld_Line` | Defect class |
| 2 | `Sink_Mark` | Defect class |
| 3 | `Flash` | Defect class |
| 4 | `Burn_Mark` | Defect class |

> ⚠️ Changing class IDs requires a coordinated update across MoldPilot, MoldTrace, and
> all deployed bundles. Treat these as immutable.
>
> When annotating, **every inspection frame that shows defects must also have a
> `Component_Base` box** around the part. Annotations with defects but no `Component_Base`
> are invalid and will degrade model performance at inference time.

---

## 7. `moldvision lake` CLI Surface

### 7.1 `lake init`

```
moldvision lake init [--root PATH]
```

Creates the folder skeleton and writes `data_lake_config.json`. If `ARIA_DATA_LAKE` is set
in the environment, that path is used as default.

---

### 7.2 `lake session import`

Registers a new qual session and its raw frames. Called after MoldTrace finishes
`extract_frames` for a session.

```
moldvision lake session import
  --session-meta  <session.json or moldtrace_meta/session.json>
  --inspection-frames <dir>             # extracted inspection-view JPEGs
  [--monitor-frames   <dir>]            # extracted HMI-view JPEGs
  [--quality-filter]                    # skip monitor frames that failed MoldTrace
                                        # monitor_quality check (default: on)
  [--overwrite]
```

Actions:
1. Copies frames into `sessions/<id>/inspection_frames/` and/or `monitor_frames/`.
2. Writes `session_meta.json` from the MoldPilot manifest fields in the provided JSON.
3. Appends `unlabeled` records to `image_index.jsonl` for every frame.

No annotation happens here. Frames are raw material.

---

### 7.3 `lake session list`

```
moldvision lake session list
  [--machine-id X] [--mold-id Y] [--part-id Z]
  [--from 2026-01-01] [--to 2026-06-01]
  [--task detect|seg]
  [--label-status labeled|unlabeled|any]
  [--marker "visible flash"]
  [--min-frames N]
```

Prints a table:

```
session_id                            machine  mold      frames  detect  seg  coverage
qual_20260324T093000Z_a1b2c3d4        mach_01  mold_a12  1 240     320   120    26%
qual_20260401T143000Z_f8e9d3c1        mach_01  mold_a12    540     540    —    100%
```

---

### 7.4 Labeling Loop: `lake label-batch create` / `commit`

This is the core labeling workflow. It replaces the ad-hoc "copy images into Label Studio,
export, manually move the COCO file" pattern with two explicit, tracked commands.

#### `lake label-batch create`

```
moldvision lake label-batch create
  --task detect|seg

  # Image selection
  --sessions s1,s2,...       # explicit list
  --all                      # all sessions with unlabeled frames
  --machine-id X             # filter
  --mold-id Y
  --marker "setup changed"
  --n 200                    # max images to include in this batch
  --seed 42                  # for reproducible sampling

  # Output
  --batch-name "batch-2026-q2-detect"
  [--batch-root PATH]        # default: data_lake/label_batches/
```

Actions:
1. Samples `--n` unlabeled images from matching sessions (stratified by session).
2. Creates `label_batches/<batch_id>/images/` with copies (or symlinks) of the frames.
3. Writes `batch_meta.json` recording task, session sources, frame list, status=`open`.
4. Prints the Label Studio import command:
   ```
   Batch created: label_batches/batch-2026-q2-detect-a3f1c2/
   Images: 200  (from 3 sessions)

   Next steps:
   1. Start the ML backend:
      moldvision label-studio-backend --task detect \
        --bundle models/defect_detection/mold-defect-v2.0.0/

   2. In Label Studio: create a project, add the ML backend, then import:
      label_batches/batch-2026-q2-detect-a3f1c2/images/

   3. After labeling, export COCO JSON to:
      label_batches/batch-2026-q2-detect-a3f1c2/export/

   4. Commit:
      moldvision lake label-batch commit --batch batch-2026-q2-detect-a3f1c2
   ```

#### `lake label-batch commit`

```
moldvision lake label-batch commit
  --batch <batch_id>
  [--coco-json PATH]          # explicit COCO file; default: batch/export/*.json
  [--dry-run]
```

Actions:
1. Reads the COCO JSON from `batch/export/`.
2. For each annotated image: writes/merges annotations into
   `sessions/<session_id>/annotations/<task>/_annotations.coco.json`.
3. Updates `image_index.jsonl` — changes `detect_status` (or `seg_status`) from
   `unlabeled` → `labeled` for committed images.
4. Sets `batch_meta.json` status to `committed`.
5. Prints a summary: images committed per session, annotations per class.

> **Partial commits are safe.** If Label Studio only annotated 150 of 200 batch images,
> the remaining 50 stay `unlabeled` in the index and can appear in the next batch.

---

### 7.5 `lake pull` — build training dataset

Assembles a training-ready `datasets/<UUID>/` from labeled images across sessions.

```
moldvision lake pull
  --task detect|seg

  # Session selection
  --sessions s1,s2,s3
  --all
  --machine-id X  --mold-id Y  --part-id Z
  --from 2026-01-01  --to 2026-06-01
  --marker "setup changed"

  # Annotation filter
  --label-status labeled            # default; only committed annotations
  --include-hard-negatives          # add hard_negatives pool (empty annotations)
  --include-backgrounds             # add backgrounds pool (empty annotations)

  # Distribution rules
  --max-per-session N               # cap contribution from any single session
  --min-per-session N               # skip sessions below this labeled-image count
  --balance-classes                 # undersample to equalise per-class annotation counts
  --min-per-class N                 # abort if any class has fewer than N samples
  --train-ratio 0.8                 # default 0.8
  --seed 42

  # Output
  --dataset-uuid UUID               # write into existing dataset, or auto-generate
  --dataset-name "my-dataset"
  --dry-run                         # print stats, create nothing
```

#### Pull algorithm

```
1. Read image_index.jsonl — filter by task, detect_status=labeled (or seg_status), and
   all --filter flags
2. Group images by session_id
3. Drop sessions below --min-per-session
4. Apply --max-per-session: random-sample within each session (seeded)
5. Collect COCO annotations from sessions/<id>/annotations/<task>/_annotations.coco.json
6. Merge COCO dicts across sessions (re-number image/annotation IDs to avoid collisions)
7. If --include-hard-negatives / --include-backgrounds: append pool images with empty anns
8. If --balance-classes: count annotations per category; undersample images of
   over-represented classes until all counts are within 2× the rarest class
9. Shuffle and split train/valid (--train-ratio, --seed)
10. Call create_dataset() + copy images into datasets/<UUID>/raw/
11. Write merged COCO JSONs to datasets/<UUID>/coco/train/ and valid/
12. Print distribution report
```

#### Dry-run distribution report

```
lake pull DRY RUN — task=detect
─────────────────────────────────────────────────────────────
Sessions included (after filters):
  qual_20260324...  mach_01  mold_a12  320 labeled → capped at 200 (--max-per-session)
  qual_20260401...  mach_01  mold_a12  540 labeled → all 540 included

Class distribution (before balance):
  Component_Base  1 520  ██████████████████████
  Weld_Line         298  ████
  Sink_Mark         187  ███
  Flash              72  █
  Burn_Mark          38  ▌
  hard_negatives     40  ▌

After --balance-classes (2× rarest = Burn_Mark × 2 = 76):
  Component_Base   76
  Weld_Line        76
  Sink_Mark        76
  Flash            72
  Burn_Mark        38

Train: 290 images / 338 annotations
Valid:  73 images /  84 annotations
─────────────────────────────────────────────────────────────
Run without --dry-run to create dataset.
```

---

### 7.6 `lake index`

```
moldvision lake index --rebuild          # full scan of sessions/ → rewrite image_index.jsonl
moldvision lake index --stats [--task detect|seg]
```

`--rebuild` is the recovery command: run it after manually copying frames, after a
`label-batch commit` error, or after any out-of-band annotation changes.

---

### 7.7 `lake models install / list / promote`

```
moldvision lake models install <bundle.mpk> --task detect|seg
moldvision lake models list    [--task detect|seg]
moldvision lake models promote <bundle_id> --channel stable|dev
```

`install` extracts the `.mpk` into `models/<task>/<bundle_id>/` and appends to `registry.json`.
`promote` updates the `active.<channel>` pointer — used to select which bundle the Label Studio
ML backend defaults to during pre-labeling.

---

## 8. Hard Negatives and Backgrounds

### 8.1 Hard Negatives

Images where a model predicted a defect that a human reviewer confirmed as a false positive.
The most impactful data for reducing false-positive rates.

**Entry workflow:**
```
moldvision lake pools add-hard-negative \
  --image sessions/<id>/inspection_frames/frame_000842.jpg \
  --reason "model_false_positive"
```
Appends to `pools/hard_negatives/manifest.jsonl` and updates the image's `detect_status`
to `hard_negative` in `image_index.jsonl`.

### 8.2 Backgrounds

Images with no component visible. Teach the model to suppress detections on empty frames.

```
moldvision lake pools add-background \
  --images sessions/<id>/inspection_frames/frame_00100*.jpg
```

Both pools are ingested as empty-annotation COCO images when `lake pull --include-hard-negatives`
or `--include-backgrounds` is used.

---

## 9. Complete End-to-End Workflow

```
① MoldPilot records a qual session
   → local: sessions/<session_id>/*.mp4 + session manifest JSON

② MoldTrace extracts frames
   python -m moldtrace run --session <uuid> --extract-frames --extract-monitor

③ Import raw frames into the data lake
   moldvision lake session import \
     --session-meta <moldtrace_path>/meta/session.json \
     --inspection-frames <moldtrace_path>/inputs/inspection_video/frames/vid_01/ \
     --monitor-frames <moldtrace_path>/inputs/process_video/frames/vid_01/

④ Review what needs labeling
   moldvision lake session list --label-status unlabeled --task detect

⑤ Create a labeling batch
   moldvision lake label-batch create \
     --task detect --mold-id mold_a12 --n 300 \
     --batch-name "mold-a12-detect-batch-01"

⑥ Start the Label Studio ML backend (pre-labeling from current model)
   moldvision label-studio-backend --task detect \
     --bundle models/defect_detection/mold-defect-v1.0.0/

⑦ Label in Label Studio
   → import label_batches/mold-a12-detect-batch-01/images/
   → ML backend pre-labels; human corrects
   → export COCO JSON to label_batches/mold-a12-detect-batch-01/export/

⑧ Commit labels back to the lake
   moldvision lake label-batch commit --batch mold-a12-detect-batch-01

⑨ Check coverage
   moldvision lake session list --task detect
   moldvision lake pull --task detect --all --dry-run   # preview distribution

⑩ Pull training dataset
   moldvision lake pull \
     --task detect --all \
     --include-hard-negatives \
     --max-per-session 250 --balance-classes --min-per-class 40 \
     --train-ratio 0.85 --dataset-name "mold-defect-2026-q2"

⑪ Train (unchanged workflow)
   moldvision dataset validate -d <UUID> --task detect
   moldvision train -d <UUID> --task detect --epochs 60 --batch-size 4

⑫ Export and bundle
   moldvision export -d <UUID> -w checkpoint_best_total.pth --format onnx_fp16
   moldvision bundle -d <UUID> -w checkpoint_best_total.pth \
     --model-name mold-defect --model-version 2.1.0 --mpk

⑬ Register and promote
   moldvision lake models install datasets/<UUID>/deploy/mold-defect-v2.1.0.mpk --task detect
   moldvision lake models promote mold-defect-v2.1.0 --channel stable

⑭ Deploy
   Copy mold-defect-v2.1.0.mpk to MoldPilot and/or MoldTrace model registry
```

---

## 10. Remote Storage — Future Migration Path

The data lake is designed so only the **storage backend changes** when moving to the cloud.
All CLI commands, labeling workflows, and training pipelines stay identical.

### 10.1 Why Move to Remote Storage?

| Trigger | Reason |
|---------|--------|
| Second annotator or remote team | Shared access to frames and annotations |
| MoldTrace running on AWS | Frames extracted on EC2; importing to local adds a full round-trip |
| Backup / disaster recovery | Local NAS insufficient for irreplaceable labeling work |
| Dataset > 50 GB | Outgrows a single workstation |

### 10.2 Backend Options — Cost First

| Option | Storage /GB/mo | Egress /GB | S3 API | Notes |
|--------|---------------|------------|:---:|-------|
| **Local / NAS** | HW only | Free | — | Phase 0. Zero recurring cost. |
| **MinIO (self-hosted)** | HW only | Free | ✅ | NAS or internal server. S3-compatible. Scale to any size. |
| **Backblaze B2** | $0.006 | $0.01 (free to CDN) | ✅ | ~4× cheaper than S3 for storage. Good first cloud option. |
| **Cloudflare R2** | $0.015 | **$0.00** | ✅ | Zero egress. Best when MoldPilot and MoldTrace pull bundles frequently. |
| **AWS S3** | $0.023 | $0.09 | ✅ | Most expensive. Justified only if the team is already on AWS for MoldTrace. |
| **DagsHub free tier** | 5 GB free | Free | via DVC | Useful to prototype DVC versioning at zero cost before committing to a backend. |

**Recommended progression:**
1. **Now** — local filesystem (`C:\ARIA\DataLake` or NAS share).
2. **When a second person needs access** — MinIO on a NAS (free, S3 API, no internet required).
3. **When MoldTrace moves to AWS** — Cloudflare R2 (zero egress prevents bill surprises when
   MoldPilot polls for bundle updates).

### 10.3 Abstraction Layer Design

```python
class ILakeStorage(Protocol):
    def exists(self, rel_path: str) -> bool: ...
    def read_bytes(self, rel_path: str) -> bytes: ...
    def write_bytes(self, rel_path: str, data: bytes) -> None: ...
    def list_prefix(self, prefix: str) -> list[str]: ...
    def delete(self, rel_path: str) -> None: ...
```

Implementations:
- `LocalLakeStorage(root: Path)` — filesystem, active now
- `S3LakeStorage(bucket, prefix, endpoint_url, client)` — covers AWS S3, R2, B2, MinIO

All paths in `image_index.jsonl`, `batch_meta.json`, and `manifest.jsonl` are **relative**
(`rel_path`) so they are portable between backends with no rewriting.

Switching backend = one key in `data_lake_config.json`:
```json
{
  "backend":          "s3",
  "s3_bucket":        "aria-data-lake",
  "s3_prefix":        "v1/",
  "s3_endpoint_url":  "https://<account>.r2.cloudflarestorage.com"
}
```

### 10.4 DVC as an Optional Versioning Layer

If the team needs reproducible dataset snapshots (i.e., "dataset_v3 used sessions A+B+C
at annotation revision X"), DVC can version the lake on top of any S3-compatible backend:

```bash
dvc init
dvc remote add -d lake s3://aria-data-lake/dvc   # or R2 / B2 / MinIO endpoint
dvc add aria_data_lake/sessions/
dvc push
```

Git tracks which `sessions/` revision each training run used. DVC adds this incrementally —
nothing in the local lake design blocks it, and it requires no platform subscription.

### 10.5 Migration Checklist (Local → Cloudflare R2)

1. Create R2 bucket `aria-data-lake`.
2. Generate R2 API token (S3-compatible credentials → `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).
3. Update `data_lake_config.json`: `"backend": "s3"`, set `s3_endpoint_url` and `s3_bucket`.
4. Run `moldvision lake sync --upload` to push existing sessions, annotations, and models.
5. Update `scripts/import_from_moldtrace.py` to point to R2 instead of local path.
6. All `lake session import`, `lake label-batch create/commit`, `lake pull` continue unchanged.

---

## 11. Implementation Phases

### Phase 1 — Lake skeleton, session import, index

New files:
- `moldvision/lake.py` — `LakeConfig`, `LocalLakeStorage`, `image_index` helpers
- `moldvision/cli.py` — `lake` subcommand group
- `moldvision/cli_handlers.py` — `handle_lake_init`, `handle_lake_session_import`,
  `handle_lake_index`

Delivered: `lake init`, `lake session import`, `lake index --rebuild / --stats`,
`lake session list`

Acceptance test:
```powershell
moldvision lake init --root C:\dev\aria_data_lake
moldvision lake session import --session-meta session.json --inspection-frames frames/
moldvision lake index --stats
moldvision lake session list --task detect
```

### Phase 2 — Labeling loop

New files:
- `moldvision/lake_label.py` — batch creation, Label Studio import helpers, commit logic

Delivered: `lake label-batch create`, `lake label-batch commit`

Acceptance test:
```powershell
moldvision lake label-batch create --task detect --all --n 50 --batch-name smoke
# manually place a tiny COCO JSON in label_batches/smoke.../export/
moldvision lake label-batch commit --batch smoke...
moldvision lake index --stats   # detect_status should show 50 labeled
```

### Phase 3 — Pull engine

New files:
- `moldvision/lake_pull.py` — session filtering, per-session cap, class balancing, COCO merge

Delivered: `lake pull` (all flags), `lake pools add-hard-negative / add-background`

Acceptance test:
```powershell
moldvision lake pull --task detect --all --max-per-session 200 --balance-classes --dry-run
moldvision lake pull --task detect --all --max-per-session 200 --balance-classes \
  --dataset-name smoke-test
moldvision dataset validate -d <UUID> --task detect
```

### Phase 4 — Model registry

New files:
- `moldvision/lake_models.py` — `registry.json` CRUD, `install`, `list`, `promote`

Delivered: `lake models install / list / promote`

### Phase 5 — Storage backend abstraction

New files:
- `moldvision/lake_storage.py` — `ILakeStorage`, `LocalLakeStorage`, `S3LakeStorage`
- Update `moldvision/lake.py` — wire backend from `data_lake_config.json`

Delivered: `lake sync --upload / --download`

### Phase 6 — MoldTrace integration script

New file (in `ARIA_MoldTrace` repo):
- `scripts/import_session_to_lake.py` — post-pipeline hook that calls
  `moldvision lake session import` with the correct MoldTrace paths

No MoldVision changes required.

---

## 12. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARIA_DATA_LAKE` | `%LOCALAPPDATA%\ARIA\DataLake` | Data lake root |
| `ARIA_LAKE_BACKEND` | `local` | `local` or `s3`; overrides `data_lake_config.json` |
| `ARIA_LAKE_S3_BUCKET` | — | S3 / R2 / B2 bucket name |
| `ARIA_LAKE_S3_ENDPOINT` | — | Custom endpoint URL for R2 / B2 / MinIO |
| `ARIA_LAKE_S3_PREFIX` | `v1/` | Key prefix inside bucket |

---

## 13. Open Questions

| # | Question | Recommendation |
|---|----------|---------------|
| 1 | `sessions/` and `datasets/` under the same `aria_data_lake/` root? | Yes — one env var, one sync command. |
| 2 | Monitor frames: all extracted by MoldTrace, or only quality-passed ones? | Default: quality-passed (`--quality-filter` on). Override with `--no-quality-filter` in `session import`. |
| 3 | `--balance-classes`: hard undersample at pull time vs training-time class weights? | Hard undersample at pull time — deterministic and reproducible. A `--class-weights` flag for training is a future option. |
| 4 | Should `lake label-batch create` copy images or use symlinks? | Copies on Windows (symlinks require admin rights); symlinks on Linux/Mac. |
| 5 | Where do training `runs/` logs go? | `aria_data_lake/runs/<dataset_uuid>/` — keeps each dataset and its training history together. |
| 6 | Should Label Studio project setup be automated (API call) inside `label-batch create`? | Desirable in Phase 3+; Phase 2 uses printed instructions to keep dependencies minimal. |
