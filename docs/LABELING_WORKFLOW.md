# Labeling Workflow — ARIA MoldVision + Label Studio

This document covers two labeling paths:

- **ML-assisted** — a trained MoldVision model pre-labels every image before it
  reaches an annotator, reducing labeling time by 60–80 %. Use this once you
  have at least one trained model.
- **Manual** — annotators draw every box or polygon from scratch. Use this for
  your first dataset or whenever no suitable model is available yet.

Jump to [Manual Labeling (No ML Model Required)](#manual-labeling-no-ml-model-required)
if you want to skip the ML-assisted path.

---

## Concept: Active Learning Loop

```
   ┌─────────────────────────────────────────────────────────┐
   │                                                         │
   │  moldvision train ──► moldvision bundle ──► ML backend  │
   │                                                    │    │
   │   new images ──► Label Studio ◄──────────── pre-labels  │
   │                       │                                 │
   │           annotator corrects/approves                   │
   │                       │                                 │
   │           export COCO JSON                              │
   │                       │                                 │
   │   moldvision dataset ingest ──► next training run       │
   │                                                         │
   └─────────────────────────────────────────────────────────┘
```

Every training cycle produces a better model that pre-labels more accurately,
which in turn speeds up the next labeling cycle.

---

## Step 1 — Install dependencies

```powershell
pip install label-studio
pip install "aria-moldvision[label-studio]"
```

`label-studio` is the web app; `aria-moldvision[label-studio]` installs the ML
backend SDK plus ONNX Runtime, OpenCV, and Pillow.

---

## Step 2 — Create a bundle from your trained model

```powershell
moldvision bundle `
  --dataset-dir datasets/<UUID> `
  --weights datasets/<UUID>/models/checkpoint_portable.pth `
  --model-name "Surface Defect Detector" `
  --model-version 1.0.0
```

Note the output bundle path (e.g. `datasets/<UUID>/deploy/checkpoint_portable_20260329_120000Z`).

---

## Step 3 — Start the ML pre-labeling backend

**Option A — run directly with Python (recommended, simplest)**

```powershell
$env:MOLDVISION_BUNDLE_DIR = "datasets/<UUID>/deploy/checkpoint_portable_20260329_120000Z"
python -m moldvision.label_studio_backend --port 9090
```

Linux / macOS:
```bash
MOLDVISION_BUNDLE_DIR="datasets/<UUID>/deploy/checkpoint_portable_20260329_120000Z" \
    python -m moldvision.label_studio_backend --port 9090
```

**Option B — use the label-studio-ml CLI (one-time `init` required)**

```powershell
# One-time setup: creates a moldvision-backend/ project folder
label-studio-ml init moldvision-backend --script "moldvision\label_studio_backend.py:MoldVisionMLBackend"

# Start (repeat whenever you need to restart the backend)
$env:MOLDVISION_BUNDLE_DIR = "datasets/<UUID>/deploy/checkpoint_portable_20260329_120000Z"
label-studio-ml start moldvision-backend --port 9090
```

Verify it is running:
```powershell
Invoke-WebRequest http://localhost:9090/health | Select-Object -Expand Content
# Should return: {"status":"UP","model_class":"MoldVisionMLBackend"}
```

---

## Step 4 — Start Label Studio

```powershell
label-studio start
```

Label Studio runs at `http://localhost:8080`. Create an account on first launch.

---

## Step 5 — Create a Label Studio project

1. Click **Create Project**.
2. Give it a name (e.g. *Surface Defect Detection v2*).
3. Go to **Labeling Setup** → select **Object Detection with Bounding Boxes**
   (or **Semantic Segmentation with Polygons** for seg tasks).

### Detect label config (copy-paste)

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="scratch" background="#FF0000"/>
    <Label value="dent" background="#00FF00"/>
    <Label value="stain" background="#0000FF"/>
  </RectangleLabels>
</View>
```

### Seg label config (detect + polygon masks)

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="scratch"/>
    <Label value="dent"/>
  </RectangleLabels>
  <PolygonLabels name="mask" toName="image">
    <Label value="scratch"/>
    <Label value="dent"/>
  </PolygonLabels>
</View>
```

Replace label names with your actual class names. They **must match** the class
names used when training the model.

---

## Step 6 — Connect the ML backend

1. In your project, go to **Settings** → **Machine Learning**.
2. Click **Add Model**.
3. Enter URL: `http://localhost:9090`
4. Toggle **Use for interactive pre-annotations** ON.
5. Toggle **Retrieve predictions when opening task** ON.
6. Click **Validate and Save**.

---

## Step 7 — Import new images

Go to **Import** and drag-and-drop your unlabeled images (`.jpg`, `.png`, etc.).
Label Studio automatically calls the ML backend and pre-annotates every image.

You can also import via the API:
```powershell
# Get your project ID from the URL: localhost:8080/projects/<ID>/
$PROJECT_ID = "1"
$API_KEY = "your-api-key-from-settings"

# Import a folder of images
Get-ChildItem "new_images\" -Filter *.jpg | ForEach-Object {
    $json = '[{"data":{"image":"/data/local-files/?d=' + $_.FullName.Replace('\','/') + '"}}]'
    Invoke-RestMethod -Method Post `
        -Uri "http://localhost:8080/api/projects/$PROJECT_ID/import" `
        -Headers @{"Authorization"="Token $API_KEY"} `
        -Body $json `
        -ContentType "application/json"
}
```

---

## Step 8 — Review and correct pre-labels

Open the **Label Stream** in Label Studio. Each image will show the model's
bounding box predictions as starting points. Annotators:

- **Accept**: click Submit without changes if the pre-label is correct.
- **Correct**: drag/resize boxes or edit polygons to fix mistakes.
- **Delete**: remove incorrect boxes and draw from scratch if needed.

Tip: set the project score threshold in backend params if predictions are too
noisy. Re-start the backend with `--with score_threshold=0.7`.

---

## Step 9 — Export annotations to COCO JSON

1. Go to **Export** → select **COCO JSON**.
2. Download the `.zip` file.
3. Extract it — it contains `result.json` (COCO annotations) and an `images/`
   folder with symlinks or copies of your images.

---

## Step 10 — Ingest into MoldVision and retrain

```powershell
# Place your exported images + COCO JSON in the labels_inbox
# Expected layout:
#   datasets/<UUID>/labels_inbox/<split>/images/<image_files>
#   datasets/<UUID>/labels_inbox/<split>/_annotations.coco.json

moldvision dataset ingest -d datasets/<UUID>

# Validate the result
moldvision dataset validate -d datasets/<UUID> --task detect

# Retrain
moldvision train `
  --dataset-dir datasets/<UUID> `
  --epochs 80 `
  --size nano
```

After retraining, create a new bundle and restart the ML backend with the
updated weights. The next labeling cycle will benefit from the improved model.

---

## Adjusting the confidence threshold

The pre-labeling threshold defaults to the value stored in the bundle's
`postprocess.json` (`score_threshold_default`). To override it for a labeling
session without retraining:

```powershell
$env:MOLDVISION_BUNDLE_DIR = "datasets/<UUID>/deploy/<bundle>"
python -m moldvision.label_studio_backend --port 9090
```

The threshold can be overridden at runtime via the Label Studio UI: **Settings → Machine Learning → your backend → ⚙ → set `score_threshold`**.

Lower the threshold to see more (noisier) pre-labels; raise it to see fewer
but higher-confidence ones.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Backend returns empty predictions | Check `MOLDVISION_BUNDLE_DIR` is set and points to a valid bundle directory |
| Label names don't match in UI | Ensure your Label Studio label config uses exactly the same class names as your training dataset |
| `cv2` not found (seg masks missing) | Run `pip install opencv-python-headless` |
| `CUDA` provider warnings | Install `onnxruntime-gpu` instead of `onnxruntime` for GPU inference |
| Image not loading | Check that Label Studio has access to the image path; use `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` |

### Enabling local file serving in Label Studio

If your images are on disk rather than a web URL, start Label Studio with:

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "C:\path\to\your\images"
label-studio start
```

---

## Manual Labeling (No ML Model Required)

Use this workflow when you are collecting your first dataset and have no trained
model yet, or when you simply want to annotate without running an ML backend.

There are three supported paths depending on the tools you prefer.

---

### Option A — Label Studio standalone (recommended)

This is the same tool used for ML-assisted labeling, just without connecting a
model backend.

**1. Install and start Label Studio**

```powershell
pip install label-studio
label-studio start
```

Enable local file serving if your images are on disk:

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "C:\path\to\your\images"
label-studio start
```

**2. Create a project**

1. Click **Create Project** and give it a name.
2. Go to **Labeling Setup** and paste the label config for your task.

Bounding box detection:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="scratch" background="#FF0000"/>
    <Label value="dent"    background="#00FF00"/>
    <Label value="stain"   background="#0000FF"/>
  </RectangleLabels>
</View>
```

Instance segmentation (bounding boxes + polygon masks):

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="scratch"/>
    <Label value="dent"/>
  </RectangleLabels>
  <PolygonLabels name="mask" toName="image">
    <Label value="scratch"/>
    <Label value="dent"/>
  </PolygonLabels>
</View>
```

Replace label names with your actual class names — they must match what you
pass to `moldvision dataset create -c`.

**3. Import images and annotate**

Go to **Import**, drag-and-drop your images, and use the Label Stream to draw
boxes or polygons manually. Do **not** connect an ML backend.

**4. Export annotations**

Go to **Export** → **COCO JSON** → download and extract the `.zip`. You will
get a `result.json` and an `images/` folder.

**5. Ingest into MoldVision**

```powershell
# Copy the exported files into labels_inbox
# Expected layout:
#   datasets/<UUID>/labels_inbox/coco/images/<image_files>
#   datasets/<UUID>/labels_inbox/coco/_annotations.coco.json

moldvision dataset ingest -d datasets/<UUID>

# Validate before training
moldvision dataset validate -d datasets/<UUID> --task detect
```

---

### Option B — YOLO format label files

Use this if you prefer a text-based format or already have labels from another
tool (e.g. labelImg, Roboflow) in YOLO format.

Each `.txt` file must have one row per annotation:
```
<class_id> <cx> <cy> <w> <h>
```
All values are normalised to `[0, 1]`. The class order must match the order of
classes in your dataset's `METADATA.json`.

```powershell
# Place images and .txt files together
# datasets/<UUID>/labels_inbox/yolo/
#   image1.jpg
#   image1.txt
#   image2.jpg
#   image2.txt

moldvision dataset ingest -d datasets/<UUID> --yolo-task detect

# For segmentation tasks
moldvision dataset ingest -d datasets/<UUID> --yolo-task seg
```

---

### Option C — Any COCO-compatible annotation tool

Tools such as **CVAT**, **VGG Image Annotator (VIA)**, or **Roboflow** can
export COCO JSON directly. The ingest pipeline accepts any valid COCO JSON file.

1. Annotate in your preferred tool.
2. Export as **COCO JSON** (with an `images/` folder alongside the `.json`).
3. Drop everything into `datasets/<UUID>/labels_inbox/coco/`.
4. Run `moldvision dataset ingest -d datasets/<UUID>`.

---

### After manual labeling — training your first model

Once your first labeled dataset is ingested, you can train a baseline model:

```powershell
moldvision train `
  --dataset-dir datasets/<UUID> `
  --epochs 80 `
  --size nano
```

After training, create a bundle and switch to the ML-assisted workflow to
speed up future labeling rounds:

```powershell
moldvision bundle `
  --dataset-dir datasets/<UUID> `
  --weights datasets/<UUID>/models/checkpoint_portable.pth `
  --model-name "My Detector" `
  --model-version 1.0.0
```

Then follow [Step 3](#step-3--start-the-ml-pre-labeling-backend) onwards in the
ML-assisted workflow above.
