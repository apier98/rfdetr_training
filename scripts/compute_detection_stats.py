#!/usr/bin/env python3
"""Compute per-class detection statistics by matching detections to COCO GT boxes.

Simple greedy matching by IoU threshold (default 0.5). Outputs per-class TP/FP/FN and precision/recall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


def iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    denom = boxAArea + boxBArea - inter
    return inter / denom if denom > 0 else 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, help="COCO _annotations.coco.json path (ground truth)")
    p.add_argument("--dets", required=True, help="Detections JSON produced by batch_infer.py")
    p.add_argument("--iou", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    dets = json.loads(Path(args.dets).read_text(encoding="utf-8"))

    # build map image->gt boxes per class
    gt_by_image = {}
    for im in gt.get("images", []):
        gt_by_image[im["file_name"]] = []
    for ann in gt.get("annotations", []):
        img = next((i["file_name"] for i in gt.get("images", []) if i["id"]==ann["image_id"]), None)
        if img is None:
            # sometimes image_id is file name already
            img = str(ann.get("image_id"))
        bbox = ann["bbox"]
        # convert COCO bbox [x,y,w,h] -> [x1,y1,x2,y2]
        box = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        gt_by_image.setdefault(img, []).append({"box": box, "cat": ann["category_id"], "matched": False})

    # group detections by image
    dets_by_image = {}
    for d in dets:
        dets_by_image.setdefault(d["image_id"], []).append(d)

    per_class = {}

    for img, gts in gt_by_image.items():
        dets_img = dets_by_image.get(img, [])
        # sort dets by score desc
        dets_img = sorted(dets_img, key=lambda x: x.get("score", 0.0), reverse=True)

        for d in dets_img:
            best_i = -1
            best_iou = 0.0
            for i,gtann in enumerate(gts):
                if gtann["matched"]:
                    continue
                if gtann["cat"] != d.get("label_id"):
                    continue
                iouv = iou(d["bbox"], gtann["box"])
                if iouv > best_iou:
                    best_iou = iouv; best_i = i

            cls = d.get("label_id")
            pc = per_class.setdefault(cls, {"tp":0, "fp":0, "fn":0, "det_count":0})
            pc["det_count"] += 1
            if best_i >= 0 and best_iou >= args.iou:
                pc["tp"] += 1
                gts[best_i]["matched"] = True
            else:
                pc["fp"] += 1

        # remaining unmatched GTs count as FN
        for gtann in gts:
            if not gtann["matched"]:
                cls = gtann["cat"]
                pc = per_class.setdefault(cls, {"tp":0, "fp":0, "fn":0, "det_count":0})
                pc["fn"] += 1

    # compute precision/recall
    out = {}
    for cls, v in per_class.items():
        tp = v["tp"]; fp = v["fp"]; fn = v["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        out[int(cls)] = {"tp":tp, "fp":fp, "fn":fn, "precision":prec, "recall":rec, "det_count": v.get("det_count",0)}

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
