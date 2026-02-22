#!/usr/bin/env python3
"""Sample images from COCO splits and save annotated visualizations.

Writes annotated images into `<dataset>/exports/visualizations/<split>/`.

Usage example:
  python scripts/visualize_annotations.py -d datasets/<UUID> -c 10 --split train --seed 1

The script will look for COCO JSONs at `coco/train/_annotations.coco.json` and
`coco/valid/_annotations.coco.json`. If images are not present inside `coco/<split>/`,
it will fall back to `raw/`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    raise ImportError("Pillow is required. Install with: pip install pillow") from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize COCO annotations by sampling images")
    p.add_argument("--dataset-dir", "-d", required=False, help="Path to dataset UUID folder")
    p.add_argument("--dataset", "-D", dest="dataset_dir", required=False, help="Alias for --dataset-dir")
    p.add_argument("--count", "-c", type=int, default=5, help="Number of images to sample (default: 5)")
    p.add_argument("--split", choices=["train", "valid", "all"], default="train", help="Which split to sample from (default: train)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    p.add_argument("--continuous", action="store_true", help="Sample a continuous block of images in filename order")
    p.add_argument("--seq-length", type=int, help="When used with --continuous, number of consecutive images to sample (default: --count)")
    p.add_argument("--start-file", help="Optional starting file_name for continuous sampling")
    p.add_argument("--out-subdir", default="exports/visualizations", help="Output folder relative to dataset dir")
    p.add_argument("--draw-polygons", action="store_true", help="Draw segmentation polygons if present")
    return p.parse_args()


def load_coco_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_image_ann_map(coco: Dict[str, Any]) -> Dict[int, Dict]:
    images = {img["id"]: img for img in coco.get("images", [])}
    ann_map: Dict[int, List[Dict]] = {img_id: [] for img_id in images.keys()}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        if img_id in ann_map:
            ann_map[img_id].append(ann)
    # return mapping image_id -> {image:..., annotations: [...]}
    out: Dict[int, Dict] = {}
    for img_id, img in images.items():
        out[img_id] = {"image": img, "annotations": ann_map.get(img_id, [])}
    return out


def find_image_path(dataset_dir: Path, split: str, file_name: str) -> Path | None:
    # prefer coco/<split>/file_name, then fall back to raw/
    candidates = [dataset_dir / "coco" / split / file_name, dataset_dir / "raw" / file_name]
    for p in candidates:
        if p.exists():
            return p
    # last resort: search recursively under dataset_dir for matching name
    for p in dataset_dir.rglob(file_name):
        if p.is_file():
            return p
    return None


def draw_annotations(img: Image.Image, anns: List[Dict], categories: Dict[int, str], draw_polygons: bool = False) -> Image.Image:
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # color palette (repeating) - ensure class 0 is green
    palette = ["#33FF33", "#FF3333", "#3333FF", "#FF33CC", "#33FFFF", "#FFAA33"]

    def hex_to_rgb(h: str) -> tuple:
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    # work in RGBA to support semi-transparent fills
    img_rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw_base = ImageDraw.Draw(img_rgba)
    draw_overlay = ImageDraw.Draw(overlay)

    for ann in anns:
        cat_id = ann.get("category_id", 0)
        color_hex = palette[int(cat_id) % len(palette)]
        rgb = hex_to_rgb(color_hex)
        outline_color = (rgb[0], rgb[1], rgb[2], 255)

        # draw bbox if present
        bbox = ann.get("bbox")
        if bbox and len(bbox) >= 4:
            x, y, w, h = bbox[:4]
            x1, y1, x2, y2 = x, y, x + w, y + h
            # thicker outline
            draw_base.rectangle([x1, y1, x2, y2], outline=outline_color, width=4)

            # fill bbox with a slightly brighter semi-transparent color (for all classes)
            brighter = (min(255, rgb[0] + 40), min(255, rgb[1] + 40), min(255, rgb[2] + 40), 90)
            draw_overlay.rectangle([x1, y1, x2, y2], fill=brighter)

            label = categories.get(cat_id, str(cat_id))
            # robust text size calculation across Pillow versions
            try:
                if font:
                    try:
                        text_w, text_h = font.getsize(label)
                    except Exception:
                        tbbox = draw_base.textbbox((0, 0), label, font=font)
                        text_w = tbbox[2] - tbbox[0]
                        text_h = tbbox[3] - tbbox[1]
                else:
                    try:
                        tbbox = draw_base.textbbox((0, 0), label)
                        text_w = tbbox[2] - tbbox[0]
                        text_h = tbbox[3] - tbbox[1]
                    except Exception:
                        text_w, text_h = (len(label) * 6, 10)
            except Exception:
                text_w, text_h = (len(label) * 6, 10)
            # label background (solid for readability)
            label_bg_color = (rgb[0], rgb[1], rgb[2], 255)
            draw_base.rectangle([x1, max(y1 - text_h - 4, 0), x1 + text_w + 4, y1], fill=label_bg_color)
            draw_base.text((x1 + 2, max(y1 - text_h - 4, 0) + 1), label, fill=(0, 0, 0, 255), font=font)

        # draw segmentation polygons if requested
        if draw_polygons and ann.get("segmentation"):
            segs = ann.get("segmentation") or []
            for seg in segs:
                if not seg:
                    continue
                coords = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                draw_base.line(coords + [coords[0]], fill=outline_color, width=2)

    # composite overlay (semi-transparent fills) onto base
    result = Image.alpha_composite(img_rgba, overlay)
    return result.convert("RGB")


def main() -> int:
    args = parse_args()
    if not args.dataset_dir:
        print("Error: --dataset-dir (or --dataset) is required", file=sys.stderr)
        return 2
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    split = args.split
    count = args.count
    random.seed(args.seed)

    # load coco jsons
    coco_maps: Dict[int, Dict] = {}
    categories_map: Dict[int, str] = {}
    available_images: List[Dict] = []

    splits_to_read = [split] if split in ("train", "valid") else ["train", "valid"]
    for s in splits_to_read:
        json_path = dataset_dir / "coco" / s / "_annotations.coco.json"
        if not json_path.exists():
            # skip missing split
            continue
        coco = load_coco_json(json_path)
        # categories
        for c in coco.get("categories", []):
            categories_map[int(c.get("id", 0))] = c.get("name", str(c.get("id", 0)))
        im_map = build_image_ann_map(coco)
        for img_id, data in im_map.items():
            data["_split"] = s
            available_images.append(data)

    if not available_images:
        print(f"No COCO images found in {dataset_dir / 'coco'} (look for train/valid).")
        return 2

    # select images: either random sample (existing behavior) or continuous block
    if args.continuous:
        seq_len = args.seq_length or count
        # consider only images from the requested splits
        candidates = [it for it in available_images if it.get("_split") in splits_to_read]
        if not candidates:
            print(f"No images available in requested splits: {splits_to_read}.")
            return 2
        # sort by filename to get a natural ordering (works with zero-padded names)
        candidates.sort(key=lambda x: x["image"].get("file_name", ""))

        # determine start index
        start_idx = None
        if args.start_file:
            for i, it in enumerate(candidates):
                if it["image"].get("file_name") == args.start_file:
                    start_idx = i
                    break
            if start_idx is None:
                print(f"Warning: start file '{args.start_file}' not found; choosing random start.")

        if start_idx is None:
            max_start = max(0, len(candidates) - seq_len)
            start_idx = random.randint(0, max_start) if max_start > 0 else 0

        sampled = candidates[start_idx : start_idx + seq_len]
        if not sampled:
            print("No images selected for continuous sampling.")
            return 2
    else:
        sample_count = min(count, len(available_images))
        sampled = random.sample(available_images, sample_count)

    out_base = dataset_dir / args.out_subdir
    for item in sampled:
        img_entry = item["image"]
        anns = item["annotations"]
        s = item.get("_split", "train")
        img_path = find_image_path(dataset_dir, s, img_entry.get("file_name"))
        if img_path is None:
            print(f"Warning: image file not found for {img_entry.get('file_name')}; skipping.")
            continue

        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB")
            vis = draw_annotations(im_rgb, anns, categories_map, draw_polygons=args.draw_polygons)

            dst_dir = out_base / s
            dst_dir.mkdir(parents=True, exist_ok=True)
            # use original name prefixed with 'annot_' to avoid collisions
            out_name = f"annot_{img_entry.get('id'):06d}_{img_entry.get('file_name')}"
            out_path = dst_dir / out_name
            vis.save(out_path)
            print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
