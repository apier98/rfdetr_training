# Transfer Artifacts & Run Inference in Another Project

This document explains how to move trained RF-DETR artifacts (models, metadata, and class lists) to another project and run inference there. It includes prerequisites, recommended file layout, how to load different checkpoint formats, and the canonical detection output format you should use when integrating with other apps.

## Purpose
- Make it simple to reuse trained checkpoints from this repo in an external project or service.
- Describe the expected files and how to load them reliably (state_dict vs pickled models).
- Specify a small, stable JSON detection output format to use for saving or sending detection results.

## Files to transfer

Recommended: create a portable bundle folder with an embedded inference runner:
- `python -m moldvision bundle -d datasets/<UUID> -w datasets/<UUID>/models/checkpoint_best_total.pth --zip`
- Copy the resulting `datasets/<UUID>/deploy/<bundle_name>/` folder (or the `.zip`) into the other project and run `python infer.py --image ...`.
- For segmentation overlays, add `--out-image out.png` and optionally tune `--mask-thresh` / `--mask-alpha`.

Copy the following from the dataset folder you trained into (example `datasets/<UUID>/`):
- `models/checkpoint.pth` (or `best.pth`, `last.pth`) — the PyTorch checkpoint.
- `METADATA.json` — contains `class_names` and other dataset metadata.
- (optional) any `classes.txt` or plain newline file you use for class names.

Keep the same folder structure if possible, e.g.:
```
my-new-project/
  models/
    checkpoint.pth
  METADATA.json
```

## Minimal environment / Dependencies
Ensure your target project has a compatible environment. Minimal Python packages:
- Python 3.9+ (match the one used for training)
- torch (same major version used during training)
- torchvision
- rfdetr (if you intend to instantiate RF-DETR models by code rather than using pickled model objects)
- opencv-python (for webcam/display)
- pillow

Example pip install command (adjust versions as needed):

```powershell
pip install torch torchvision rfdetr opencv-python pillow
```

Tip: Use a virtual environment (venv) for reproducibility.

## Loading checkpoints: rules and examples
Checkpoints commonly come in two forms:
1. `state_dict` dict saved to `.pth` (recommended).
2. A pickled `nn.Module` object saved in the checkpoint (less portable; may require `weights_only=False` on load).

General strategy:
- Prefer to load the checkpoint as a `state_dict`. This is portable and safe.
- If the checkpoint contains a pickled model object and you trust the source, you can load it, but doing so may execute code from the checkpoint.

Example: safe load from `state_dict` and instantiate the proper model

```python
import torch
from rfdetr import RFDETRNano  # or RFDETRSmall, etc.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('models/checkpoint.pth', map_location='cpu')
# find the state dict inside the checkpoint
state = None
for key in ('model_state_dict','state_dict','model','net'):
    if isinstance(ckpt, dict) and key in ckpt:
        state = ckpt[key]
        break
if state is None and isinstance(ckpt, dict) and any(k.startswith('transformer') for k in ckpt.keys()):
    state = ckpt

# read num_classes from METADATA.json or detect from checkpoint
import json
meta = json.load(open('METADATA.json'))
num_classes = len(meta.get('class_names', []))

model = RFDETRNano(num_classes=num_classes)
model.load_state_dict(state, strict=False)
model.to(device).eval()
```

If the checkpoint contains a pickled model object (ckpt['model'] is an nn.Module):

```python
# only do this for trusted checkpoints
ck = torch.load('models/checkpoint.pth', map_location=device, weights_only=False)
if isinstance(ck, dict) and 'model' in ck and hasattr(ck['model'], 'state_dict'):
    model = ck['model']
    model.to(device).eval()
```

Notes about `num_classes` and head mismatches
- If you instantiate an off-the-shelf RFDETR model (e.g., default with 91 classes) but the checkpoint was trained for a different number of classes, you must either:
  - Instantiate with the correct `num_classes` (best), or
  - Use a loader that replaces only the classifier head to match size (the helper in this repo performs partial replacement and weight copying).
- When moving artifacts to another project, prefer instantiating with explicit `num_classes` read from `METADATA.json` so the model head matches directly.

## Running inference (CLI / programmatic)
If you copy `scripts/infer_webcam.py` into the other project or adapt the logic, the CLI supports these important flags (examples):
- `--weights` path to checkpoint
- `--size` model size (nano, small, base, medium)
- `--num-classes` optional override
- `--classes-file` path to `METADATA.json` or newline class file
- `--use-checkpoint-model` if the checkpoint contains a pickled model object and you want to use it directly
- `--device` optional (e.g. `cuda:0`)

Example CLI (PowerShell style):

```powershell
python .\scripts\infer_webcam.py \
  --camera 0 \
  --weights models\checkpoint.pth \
  --size nano \
  --num-classes 1 \
  --classes-file METADATA.json \
  --device cuda:0
```

Programmatic example (returns JSON-friendly dict):

```python
import torch
from rfdetr import RFDETRNano
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
meta = json.load(open('METADATA.json'))
num_classes = len(meta.get('class_names', []))
model = RFDETRNano(num_classes=num_classes).to(device).eval()
ckpt = torch.load('models/checkpoint.pth', map_location=device)
# pick state or model as shown in earlier examples
state = ckpt.get('model_state_dict', ckpt.get('state_dict', None)) or ckpt
model.load_state_dict(state, strict=False)

# run on an input tensor `img_tensor` shaped [1,C,H,W]
with torch.no_grad():
    out = model(img_tensor)
# parse `out` into boxes/scores/labels as described below
```

## Detection output format (canonical JSON)
Use this compact JSON-friendly format to return or save detections from any inference endpoint.

Schema (per inference call / per image):

- `image_id` (optional): string or int identifier for the input image
- `detections`: list of detection objects
  - each detection object:
    - `bbox`: [x1, y1, x2, y2] (absolute pixel coordinates, top-left / bottom-right)
    - `score`: float (0.0 - 1.0)
    - `label_id`: int (class index, 0-based)
    - `label_name`: string (human-friendly name if available)

Example JSON payload (pretty-printed):

```json
{
  "image_id": "frame_0001",
  "detections": [
    {"bbox": [12.3, 45.0, 90.2, 130.6], "score": 0.87, "label_id": 0, "label_name": "placca"},
    {"bbox": [200.1, 30.0, 260.5, 95.2], "score": 0.45, "label_id": 0, "label_name": "placca"}
  ]
}
```

Why this format?
- Simple to serialize using `json.dumps()`.
- Coordinates are absolute pixel coordinates; many consumers expect that.
- Includes both `label_id` and `label_name` to ease integration with UIs.

How to convert common output types into this JSON format
- If your model returns a dict with `boxes`, `scores`, `labels` (tensors), convert by pulling them to CPU and iterating per detection, filtering by a confidence threshold.
- If your model returns `supervision.Detections`, you can access `xyxy`, `confidence` and `class_id` fields and map them to the JSON schema.

Small conversion snippet:

```python
import numpy as np

def detections_to_json(boxes, scores, labels, class_names=None, image_id=None, score_thresh=0.3):
    outs = { 'image_id': image_id, 'detections': [] }
    for i in range(len(boxes)):
        sc = float(scores[i])
        if sc < score_thresh:
            continue
        b = [float(x) for x in boxes[i]]
        lid = int(labels[i])
        lname = (class_names[lid] if class_names and 0 <= lid < len(class_names) else str(lid))
        outs['detections'].append({ 'bbox': b, 'score': sc, 'label_id': lid, 'label_name': lname })
    return outs
```

## Practical tips and gotchas
- Always pass `--num-classes` or read `METADATA.json` so the instantiated model head matches your checkpoint.
- If you see device mismatch errors (CPU vs CUDA), make sure you call `.to(device)` on the instantiated model and on any replacement modules before inference. The loader in this repo does this for you.
- If the checkpoint is a pickled model object, it may contain custom classes; ensure those classes are importable in the target project or avoid untrusted pickles.
- For batch or server inference, adapt the webcam code into a function that takes a numpy image or bytes and returns the JSON schema above.

## Example: HTTP inference endpoint (Flask sketch)

```python
from flask import Flask, request, jsonify
import torch
import json

app = Flask(__name__)
# load model at startup (see earlier examples)

@app.route('/infer', methods=['POST'])
def infer():
    # receive an image via form-data or base64 and convert to tensor
    img_tensor = preprocess_request_to_tensor(request)
    with torch.no_grad():
        out = model(img_tensor)
    boxes, scores, labels = parse_out_to_arrays(out)
    resp = detections_to_json(boxes, scores, labels, class_names=meta['class_names'], image_id=request.args.get('id', None))
    return jsonify(resp)
```

## Summary
- Copy `models/checkpoint.pth` and `METADATA.json` to the new project.
- Install required packages and instantiate the RFDETR model using `num_classes` from `METADATA.json`.
- Prefer `state_dict` checkpoints; if using pickled model objects, only load trusted checkpoints.
- Use the provided canonical JSON schema for detection outputs to make downstream integration predictable and stable.

If you want, I can:
- Add a small `requirements.txt` snippet for this repo.
- Add a helper function file `infer_helpers.py` that encapsulates robust checkpoint loading and JSON conversion for easier copy/paste into other projects.

Which of these would you like next?
