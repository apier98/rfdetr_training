"""Label Studio ML backend for ARIA MoldVision.

Provides pre-labeling (bounding boxes and segmentation masks) to Label Studio
using a MoldVision deployment bundle. Supports both detect and seg tasks.

Usage
-----
Install dependencies::

    pip install "aria-moldvision[label-studio]"

Start the backend::

    MOLDVISION_BUNDLE_DIR=/path/to/bundle label-studio-ml start moldvision/label_studio_backend.py -p 9090

Or with an explicit bundle dir flag::

    label-studio-ml start moldvision/label_studio_backend.py \\
        --with bundle_dir=/path/to/bundle --port 9090

Then in Label Studio → Project Settings → Machine Learning → Add Model → http://localhost:9090

Label Studio project configuration
------------------------------------
For detect bundles, use::

    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="defect_a"/>
        <Label value="defect_b"/>
      </RectangleLabels>
    </View>

For seg bundles, add a PolygonLabels tag::

    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="defect_a"/>
      </RectangleLabels>
      <PolygonLabels name="mask" toName="image">
        <Label value="defect_a"/>
      </PolygonLabels>
    </View>

The backend reads the project's label config and automatically selects the
correct tag names; no hardcoding is necessary.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

try:
    import onnxruntime as ort
    _ORT_OK = True
except ImportError:
    _ORT_OK = False

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

try:
    from PIL import Image
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    from label_studio_ml.model import LabelStudioMLBase
    _LS_OK = True
except ImportError:
    _LS_OK = False
    # Fallback stub so the module can be imported even without the SDK installed
    class LabelStudioMLBase:  # type: ignore[no-redef]
        def predict(self, tasks, context=None, **kwargs):
            raise RuntimeError("label-studio-ml package not installed. Run: pip install label-studio-ml")
        def get_local_path(self, url, task_id=None):
            raise RuntimeError("label-studio-ml package not installed.")

from .jsonutil import load_json

# Environment variable used to pass the bundle directory to the backend process.
BUNDLE_DIR_ENV = "MOLDVISION_BUNDLE_DIR"


# ---------------------------------------------------------------------------
# Internal ONNX runner — mirrors OnnxInferenceService in ARIA_MoldPilot
# ---------------------------------------------------------------------------

class _OnnxBundleRunner:
    """Loads a MoldVision bundle and runs ONNX inference with proper letterbox preprocessing."""

    def __init__(self, bundle_dir: Path) -> None:
        if not _NUMPY_OK or not _ORT_OK or not _PIL_OK:
            raise RuntimeError("numpy, onnxruntime, and Pillow are required. Run: pip install numpy onnxruntime Pillow")

        self.bundle_dir = bundle_dir
        self.preprocess_cfg: dict[str, Any] = load_json(bundle_dir / "preprocess.json")
        self.postprocess_cfg: dict[str, Any] = load_json(bundle_dir / "postprocess.json")

        # Load class names — prefer manifest.json inline map, fall back to classes.json list.
        try:
            manifest = load_json(bundle_dir / "manifest.json")
            classes_map: dict = manifest.get("classes", {})
            self.class_names = [classes_map[str(i)] for i in range(len(classes_map))]
        except Exception:
            try:
                raw = load_json(bundle_dir / "classes.json")
                self.class_names = list(raw) if isinstance(raw, list) else []
            except Exception:
                self.class_names = []

        # Prefer quantized → fp16 → fp32 model.
        model_path: Optional[Path] = None
        for candidate in ("model_quantized.onnx", "model_fp16.onnx", "model.onnx"):
            p = bundle_dir / candidate
            if p.exists():
                model_path = p
                break
        if model_path is None:
            raise FileNotFoundError(f"No ONNX model found in bundle: {bundle_dir}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        _log.info("Loaded %s on %s", model_path.name, providers[0])

        input_meta = self.session.get_inputs()[0]
        _, _, self.target_h, self.target_w = input_meta.shape

        pre = self.preprocess_cfg
        self.mean = np.array(pre.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(pre.get("normalize", {}).get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

        post = self.postprocess_cfg
        self.score_threshold = float(post.get("score_threshold_default", 0.5))
        self.topk = int(post.get("topk_default", 100))
        self.nms_iou = float(post.get("nms_iou_threshold_default", 0.3))

    # ------------------------------------------------------------------
    # Pre / postprocess
    # ------------------------------------------------------------------

    def _preprocess(self, pil_image: "Image.Image") -> tuple[np.ndarray, dict]:
        """Letterbox resize + ImageNet normalise → NCHW float32 tensor."""
        w, h = pil_image.width, pil_image.height
        tw, th = self.target_w, self.target_h
        scale = min(tw / w, th / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        pad_left = (tw - new_w) // 2
        pad_top = (th - new_h) // 2

        canvas = Image.new("RGB", (tw, th), (114, 114, 114))
        canvas.paste(pil_image.resize((new_w, new_h), Image.BILINEAR), (pad_left, pad_top))

        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        tensor = arr.transpose(2, 0, 1)[None].astype(np.float32)

        meta = {
            "scale": scale, "pad_left": pad_left, "pad_top": pad_top,
            "orig_w": w, "orig_h": h,
        }
        return tensor, meta

    def _unletterbox_xyxy(self, x1: float, y1: float, x2: float, y2: float, meta: dict) -> tuple[float, float, float, float]:
        scale = meta["scale"]
        pl, pt = meta["pad_left"], meta["pad_top"]
        ow, oh = meta["orig_w"], meta["orig_h"]
        x1 = max(0.0, min((x1 - pl) / scale, float(ow)))
        y1 = max(0.0, min((y1 - pt) / scale, float(oh)))
        x2 = max(0.0, min((x2 - pl) / scale, float(ow)))
        y2 = max(0.0, min((y2 - pt) / scale, float(oh)))
        return x1, y1, x2, y2

    def _nms(self, detections: list[dict]) -> list[dict]:
        order = list(range(len(detections)))
        kept: list[dict] = []
        while order:
            i = order.pop(0)
            kept.append(detections[i])
            x1, y1, w1, h1 = detections[i]["bbox_pct"]
            remaining: list[int] = []
            for j in order:
                x2, y2, w2, h2 = detections[j]["bbox_pct"]
                ix1, iy1 = max(x1, x2), max(y1, y2)
                ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = w1 * h1 + w2 * h2 - inter
                iou = inter / union if union > 0 else 0.0
                if iou <= self.nms_iou:
                    remaining.append(j)
            order = remaining
        return kept

    def _mask_to_polygon_pct(self, mask_hw: np.ndarray, orig_w: int, orig_h: int) -> Optional[list[list[float]]]:
        """Convert a binary HxW mask to a list of [x_pct, y_pct] polygon points."""
        if not _CV2_OK:
            return None
        mask_u8 = (mask_hw > 0.5).astype(np.uint8)
        if mask_u8.sum() == 0:
            return None
        # Resize mask to original image dimensions before finding contours.
        mask_resized = cv2.resize(mask_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Keep only the largest contour.
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 4:
            return None
        pts = contour.reshape(-1, 2)
        return [[float(p[0]) / orig_w * 100.0, float(p[1]) / orig_h * 100.0] for p in pts]

    def run(self, pil_image: "Image.Image") -> list[dict]:
        """Run inference and return a list of detections.

        Each detection is a dict with::

            {
                "label": str,
                "score": float,
                "bbox_pct": (x, y, w, h),   # percentage of image dimensions
                "mask_polygon": [[x_pct, y_pct], ...] or None,
            }
        """
        tensor, meta = self._preprocess(pil_image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: tensor})
        output_map = {self.session.get_outputs()[i].name: outputs[i] for i in range(len(outputs))}

        pred_logits: Optional[np.ndarray] = output_map.get("pred_logits")
        pred_boxes: Optional[np.ndarray] = output_map.get("pred_boxes")
        pred_masks: Optional[np.ndarray] = output_map.get("pred_masks")

        if pred_logits is None or pred_boxes is None:
            _log.warning("Model output missing pred_logits or pred_boxes. Got: %s", list(output_map.keys()))
            return []

        logits = pred_logits[0]   # (num_queries, num_classes)
        boxes = pred_boxes[0]     # (num_queries, 4) cxcywh normalised to model input
        num_classes = logits.shape[-1]
        orig_w, orig_h = meta["orig_w"], meta["orig_h"]

        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        flat = probs.reshape(-1)
        k = min(self.topk, flat.size)
        top_idx = np.argpartition(flat, -k)[-k:]
        top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

        detections: list[dict] = []
        for idx in top_idx:
            score = float(flat[idx])
            if score < self.score_threshold:
                break
            class_id = int(idx % num_classes)
            query_idx = int(idx // num_classes)
            cx, cy, bw, bh = (float(v) for v in boxes[query_idx])
            # cxcywh normalised → xyxy in model-space pixels
            cx_px = cx * self.target_w
            cy_px = cy * self.target_h
            bw_px = bw * self.target_w
            bh_px = bh * self.target_h
            x1, y1, x2, y2 = self._unletterbox_xyxy(
                cx_px - bw_px / 2, cy_px - bh_px / 2,
                cx_px + bw_px / 2, cy_px + bh_px / 2,
                meta,
            )

            mask_polygon: Optional[list[list[float]]] = None
            if pred_masks is not None:
                mask_hw = pred_masks[0, query_idx]  # (H, W)
                mask_polygon = self._mask_to_polygon_pct(mask_hw, orig_w, orig_h)

            label = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            detections.append({
                "label": label,
                "score": score,
                "bbox_pct": (
                    float(x1) / orig_w * 100.0,
                    float(y1) / orig_h * 100.0,
                    float(x2 - x1) / orig_w * 100.0,
                    float(y2 - y1) / orig_h * 100.0,
                ),
                "mask_polygon": mask_polygon,
            })

        return self._nms(detections) if detections else detections


# ---------------------------------------------------------------------------
# Label Studio ML backend class
# ---------------------------------------------------------------------------

class MoldVisionMLBackend(LabelStudioMLBase):
    """Label Studio ML backend that pre-labels images using a MoldVision bundle.

    Configuration (via ``--with`` flags or environment variables)::

        bundle_dir   Path to the MoldVision bundle directory.
                     Overridden by MOLDVISION_BUNDLE_DIR env var.
        score_threshold  Override the bundle's default score threshold (0-1).

    The backend automatically detects whether the loaded model is a detect or
    seg model from its ONNX outputs, and produces the appropriate Label Studio
    annotation types (``rectanglelabels`` and/or ``polygonlabels``).
    """

    def setup(self) -> None:
        bundle_dir_str: str = (
            os.environ.get(BUNDLE_DIR_ENV, "")
            or self.get("bundle_dir") or ""  # type: ignore[attr-defined]
        )
        if not bundle_dir_str:
            raise RuntimeError(
                f"Bundle directory not configured. Set the {BUNDLE_DIR_ENV!r} environment variable "
                "or pass --with bundle_dir=/path/to/bundle when starting the backend."
            )
        self._runner = _OnnxBundleRunner(Path(bundle_dir_str))

        # Allow score threshold override from backend params.
        override = self.get("score_threshold")  # type: ignore[attr-defined]
        if override is not None:
            try:
                self._runner.score_threshold = float(override)
            except (ValueError, TypeError):
                pass

        _log.info(
            "MoldVisionMLBackend ready — bundle: %s — classes: %s — threshold: %.2f",
            bundle_dir_str,
            self._runner.class_names,
            self._runner.score_threshold,
        )

    def predict(self, tasks: list[dict], context: Any = None, **kwargs: Any) -> list[dict]:
        """Return pre-annotations for each task in Label Studio format."""
        # Discover tag names from the project's label configuration once per call.
        rect_from, rect_to = self._find_control_tag("RectangleLabels", fallback_from="label", fallback_to="image")
        poly_from, poly_to = self._find_control_tag("PolygonLabels", fallback_from="mask", fallback_to="image")

        results: list[dict] = []
        for task in tasks:
            image_url: Optional[str] = task.get("data", {}).get("image")
            if not image_url:
                _log.warning("Task %s has no 'image' field in data; skipping.", task.get("id"))
                results.append({"result": [], "score": 0.0})
                continue

            try:
                image_path = self.get_local_path(image_url, task_id=task.get("id"))  # type: ignore[attr-defined]
                pil_image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                _log.error("Failed to load image for task %s: %s", task.get("id"), exc)
                results.append({"result": [], "score": 0.0})
                continue

            detections = self._runner.run(pil_image)

            annotations: list[dict] = []
            for det in detections:
                x, y, w, h = det["bbox_pct"]
                label = det["label"]
                score = det["score"]

                annotations.append({
                    "from_name": rect_from,
                    "to_name": rect_to,
                    "type": "rectanglelabels",
                    "value": {
                        "x": x, "y": y, "width": w, "height": h,
                        "rectanglelabels": [label],
                    },
                    "score": score,
                })

                if det["mask_polygon"] is not None:
                    annotations.append({
                        "from_name": poly_from,
                        "to_name": poly_to,
                        "type": "polygonlabels",
                        "value": {
                            "points": det["mask_polygon"],
                            "polygonlabels": [label],
                        },
                        "score": score,
                    })

            avg_score = float(np.mean([d["score"] for d in detections])) if detections else 0.0
            results.append({"result": annotations, "score": avg_score})

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_control_tag(
        self,
        tag_type: str,
        *,
        fallback_from: str,
        fallback_to: str,
    ) -> tuple[str, str]:
        """Scan the project's label config for a control tag of *tag_type*.

        Returns ``(from_name, to_name)`` if found, otherwise falls back to the
        provided defaults. This allows the backend to work with any Label Studio
        project configuration without hardcoding tag names.
        """
        try:
            parsed: dict = self.parsed_label_config  # type: ignore[attr-defined]
            for name, info in parsed.items():
                if info.get("type", "").lower() == tag_type.lower():
                    to_names: list[str] = info.get("to_name", [])
                    to_name = to_names[0] if to_names else fallback_to
                    return name, to_name
        except Exception:
            pass
        return fallback_from, fallback_to
