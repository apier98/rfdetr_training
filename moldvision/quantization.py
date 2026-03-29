from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from .pathutil import resolve_path

import numpy as np
from PIL import Image

try:
    import onnxruntime.quantization as oq  # type: ignore
except ImportError:
    oq = None


def _letterbox_to_array(img: Image.Image, target_h: int, target_w: int) -> np.ndarray:
    """Resize with letterboxing (pad to square), normalize with ImageNet stats.

    Matches RF-DETR's inference preprocessing so calibration activations are
    representative of real deployment inputs.
    """
    iw, ih = img.size
    scale = min(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = img.resize((nw, nh), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    pad_x = (target_w - nw) // 2
    pad_y = (target_h - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return (arr - mean) / std


class CalibrationDataReader:
    """ONNX Runtime Calibration Data Reader for COCO images."""

    def __init__(
        self,
        *,
        image_paths: Sequence[Path],
        input_name: str,
        target_h: int,
        target_w: int,
        batch_size: int = 1,
    ):
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.target_h = target_h
        self.target_w = target_w
        self.batch_size = max(1, int(batch_size))
        self._ptr = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._ptr >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[self._ptr : self._ptr + self.batch_size]
        self._ptr += self.batch_size

        batch_data = []
        for p in batch_paths:
            try:
                img = Image.open(str(p)).convert("RGB")
                arr = _letterbox_to_array(img, self.target_h, self.target_w)
                batch_data.append(arr)
            except Exception as e:
                print(f"Warning: Calibration skipped corrupted image {p}: {e}")
                continue

        if not batch_data:
            return self.get_next()

        data = np.stack(batch_data, axis=0).astype(np.float32)
        return {self.input_name: data}

    def rewind(self):
        self._ptr = 0


def quantize_onnx_model(
    *,
    model_path: Path,
    output_path: Path,
    calibration_data: Optional[Sequence[Path]] = None,
    target_h: int = 640,
    target_w: int = 640,
    opset: int = 18,
    verbose: bool = False,
    seed: int = 42,
) -> bool:
    """Quantize an ONNX model to INT8 (static or dynamic)."""
    if oq is None:
        raise ImportError("onnxruntime-quantization is required for quantization.")

    model_path = resolve_path(model_path)
    output_path = resolve_path(output_path)

    if calibration_data and len(calibration_data) > 0:
        if verbose:
            print(f"Performing STATIC quantization with {len(calibration_data)} images.")

        # Shuffle so calibration covers the class distribution, not just the
        # first N images (which may be sorted by class or capture time).
        shuffled = list(calibration_data)
        random.Random(seed).shuffle(shuffled)

        import onnx  # type: ignore
        model = onnx.load(str(model_path))
        input_name = model.graph.input[0].name

        dr = CalibrationDataReader(
            image_paths=shuffled,
            input_name=input_name,
            target_h=target_h,
            target_w=target_w,
        )

        oq.quantize_static(
            model_input=str(model_path),
            model_output=str(output_path),
            calibration_data_reader=dr,
            op_types_to_quantize=None,  # all supported op types
            per_channel=True,
            reduce_range=False,
            activation_type=oq.QuantType.QUInt8,
            weight_type=oq.QuantType.QInt8,
        )
    else:
        if verbose:
            print("Performing DYNAMIC quantization (no calibration data).")
        # quantize_dynamic only supports weight_type; activation quantization
        # happens at runtime and cannot be specified here.
        oq.quantize_dynamic(
            model_input=str(model_path),
            model_output=str(output_path),
            per_channel=True,
            reduce_range=False,
            weight_type=oq.QuantType.QInt8,
        )

    return output_path.exists()

