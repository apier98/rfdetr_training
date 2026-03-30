#!/usr/bin/env python3
"""Run inference over all images in a COCO split, loading the model once.

Writes a single COCO-like list of detection dicts to `--out-json` and optional
overlay images to `--out-dir`.

Supports --backend pytorch (default) or --backend onnx with --onnx-model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from infer_helpers import instantiate_model, load_checkpoint_weights, parse_model_output
from infer_webcam import run_inference, preprocess_frame_to_tensor, load_onnx_session, onnx_input_hw, run_onnx_frame


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--weights", default=None, help="path to model weights (.pth); required when --backend pytorch")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-dir", default=None, help="optional dir to save overlay images")
    p.add_argument("--task", choices=("detect","seg"), default="detect")
    p.add_argument("--size", choices=("nano","small","base","medium"), default="nano")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--use-checkpoint-model", action="store_true")
    p.add_argument("--classes-file", default=None)
    # ONNX backend
    p.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch",
                   help="inference backend: pytorch (default) or onnx")
    p.add_argument("--onnx-model", type=str, default=None,
                   help="path to ONNX model file (required when --backend onnx)")
    return p.parse_args()


def load_class_names(path: str):
    if not path:
        return []
    try:
        j = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(j, dict) and "class_names" in j:
            return list(j["class_names"])
    except Exception:
        pass
    return []


def main():
    args = parse_args()
    imgs = sorted(Path(args.images_dir).glob("*.jpg"))
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(args.classes_file)

    if args.backend == "onnx":
        if not args.onnx_model:
            print("Error: --onnx-model is required when --backend onnx")
            return
        session = load_onnx_session(args.onnx_model, None)
        onnx_target_h, onnx_target_w = onnx_input_hw(session)
        model = None
        module_for_torch = None
    else:
        if not args.weights:
            print("Error: --weights is required when --backend pytorch")
            return
        model = instantiate_model(size=args.size, num_classes=args.num_classes, task=args.task)
        ok, replacement = load_checkpoint_weights(model, args.weights, device, allow_replace_model=args.use_checkpoint_model, verbose=False)
        if replacement is not None:
            model = replacement
        try:
            model.to(device)
        except Exception:
            pass
        try:
            model.eval()
        except Exception:
            pass

        module_for_torch = None
        if hasattr(model, "to") and callable(getattr(model, "to")):
            module_for_torch = model
        else:
            for attr in ("model","net","network","module","detector","backbone"):
                inner = getattr(model, attr, None)
                if inner is not None and hasattr(inner, "to"):
                    module_for_torch = inner
                    break

    detections = []

    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]

        if args.backend == "onnx":
            boxes, scores, labels, masks = run_onnx_frame(
                session, img, onnx_target_h, onnx_target_w,
                args.threshold, args.task == "seg", 0.5,
            )
        else:
            tensor = preprocess_frame_to_tensor(img, device)
            with torch.no_grad():
                out = None
                try:
                    out = run_inference(model, module_for_torch, tensor)
                except Exception as e:
                    # try unbatched / pil fallback
                    try:
                        t2 = tensor.squeeze(0)
                        out = run_inference(model, module_for_torch, t2)
                    except Exception:
                        try:
                            from infer_webcam import tensor_to_pil
                            pil = tensor_to_pil(tensor)
                            out = run_inference(model, module_for_torch, pil)
                        except Exception:
                            out = None

            parsed = parse_model_output(out, w, h, score_thresh=args.threshold, return_masks=(args.task=="seg"))
            if isinstance(parsed, tuple) and len(parsed) == 4:
                boxes, scores, labels, masks = parsed
            else:
                boxes, scores, labels = parsed
                masks = None

        # save overlay if requested
        if out_dir is not None:
            disp = img.copy()
            try:
                from infer_image import overlay_masks, draw_detections
                if masks:
                    overlay_masks(disp, masks, labels, alpha=0.45)
                draw_detections(disp, boxes, scores, labels, class_names)
            except Exception:
                pass
            cv2.imwrite(str(out_dir / p.name), disp)

        # append detections
        for b, s, l in zip(boxes, scores, labels):
            detections.append({"image_id": p.name, "bbox": [float(x) for x in b], "score": float(s), "label_id": int(l), "label_name": (class_names[int(l)] if class_names and 0<=int(l)<len(class_names) else str(int(l)) )})

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(detections, indent=2), encoding="utf-8")
    print(f"Wrote {len(detections)} detections to {args.out_json}")


if __name__ == "__main__":
    main()
