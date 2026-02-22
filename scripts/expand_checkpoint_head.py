#!/usr/bin/env python3
"""Expand classifier heads in a checkpoint by inserting new class outputs.

This is a heuristic: it copies an existing class's weights into the new slot so the
model can output the additional class without retraining. Use with caution — for
best results, finetune after expansion.

Example:
  python scripts/expand_checkpoint_head.py \
    --in datasets/<UUID>/models/checkpoint.pth \
    --out datasets/<UUID>/models/checkpoint.pth.expanded3.pth \
    --new-classes 3 --copy-from 0 --insert-at 0
"""

from __future__ import annotations
import argparse
import torch
from pathlib import Path
import json


def find_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in ckpt:
                return ckpt[key]
        # fallback: if dict looks like state_dict (contains class_embed keys)
        for k in ckpt.keys():
            if k.endswith("class_embed.weight") or "class_embed" in k:
                return ckpt
    return None


def expand_tensor(tensor: torch.Tensor, oldC: int, newC: int, copy_from: int = 0, insert_at: int = 0):
    # expand along the first dimension when it equals oldC, else try last dim
    if tensor.dim() >= 1 and tensor.shape[0] == oldC:
        out = torch.zeros((newC, ) + tensor.shape[1:], dtype=tensor.dtype)
        # fill by copying ranges
        if insert_at == 0:
            out[0:insert_at] = out[0:insert_at]
        # copy before insert
        if insert_at > 0:
            out[0:insert_at] = tensor[0:insert_at]
        # copy inserted slot
        out[insert_at] = tensor[copy_from]
        # copy remaining
        out[insert_at+1:] = tensor[insert_at:]
        return out
    # try last dim
    if tensor.dim() >= 1 and tensor.shape[-1] == oldC:
        out_shape = list(tensor.shape)
        out_shape[-1] = newC
        out = torch.zeros(out_shape, dtype=tensor.dtype)
        # copy slices
        # copy up to insert
        if insert_at > 0:
            slicer_src = [slice(None)] * tensor.dim()
            slicer_dst = [slice(None)] * tensor.dim()
            slicer_src[-1] = slice(0, insert_at)
            slicer_dst[-1] = slice(0, insert_at)
            out[tuple(slicer_dst)] = tensor[tuple(slicer_src)]
        # copy inserted
        slicer_dst = [slice(None)] * tensor.dim(); slicer_dst[-1] = insert_at
        slicer_src = [slice(None)] * tensor.dim(); slicer_src[-1] = copy_from
        out[tuple(slicer_dst)] = tensor[tuple(slicer_src)]
        # copy remaining
        slicer_src = [slice(None)] * tensor.dim(); slicer_src[-1] = slice(insert_at, oldC)
        slicer_dst = [slice(None)] * tensor.dim(); slicer_dst[-1] = slice(insert_at+1, newC)
        out[tuple(slicer_dst)] = tensor[tuple(slicer_src)]
        return out
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    p.add_argument("--new-classes", type=int, required=True)
    p.add_argument("--copy-from", type=int, default=0, help="index of existing class to copy weights from")
    p.add_argument("--insert-at", type=int, default=0, help="position to insert new class (0..new-1)")
    args = p.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    if not infile.exists():
        print("Input checkpoint not found:", infile)
        return 2

    ckpt = torch.load(str(infile), map_location="cpu")
    state = find_state_dict(ckpt)
    if state is None:
        print("Could not find state_dict in checkpoint. Aborting.")
        return 3

    if isinstance(state, dict) and all(isinstance(v, torch.Tensor) for v in state.values()):
        # detect old class count by examining a class_embed.weight key
        sample_key = None
        for k in state.keys():
            if k.endswith("class_embed.weight"):
                sample_key = k; break
        if sample_key is None:
            # try any key that endswith class_embed.bias
            for k in state.keys():
                if k.endswith("class_embed.bias"):
                    sample_key = k.replace("bias", "weight"); break

        if sample_key is None:
            print("Could not find class_embed weight key in state_dict. Aborting.")
            return 4

        oldC = state[sample_key].shape[0]
        newC = args.new_classes
        if newC <= oldC:
            print(f"New classes ({newC}) must be > existing classes ({oldC}).")
            return 5

        modified = False
        new_state = dict(state)
        for k, v in state.items():
            if not isinstance(v, torch.Tensor):
                continue
            # attempt expansion if one of the dims matches oldC
            try:
                expanded = expand_tensor(v, oldC, newC, copy_from=args.copy_from, insert_at=args.insert_at)
            except Exception:
                expanded = None
            if expanded is not None:
                new_state[k] = expanded
                modified = True

        if not modified:
            print("No tensors were expanded. Check the checkpoint structure and keys.")
            return 6

        # save
        out_ckpt = dict(ckpt) if isinstance(ckpt, dict) else {}
        # replace state in same key if present
        replaced = False
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in out_ckpt:
                out_ckpt[key] = new_state
                replaced = True
                break
        if not replaced:
            out_ckpt = new_state

        torch.save(out_ckpt, str(outfile))
        # write metadata
        meta = {
            "expanded_from": str(infile.name),
            "old_num_classes": oldC,
            "new_num_classes": newC,
            "copy_from": args.copy_from,
            "insert_at": args.insert_at,
        }
        try:
            Path(outfile.parent / (outfile.name + ".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf8")
        except Exception:
            pass
        print(f"Wrote expanded checkpoint to {outfile}")
        return 0

    else:
        print("Unsupported checkpoint format (expected state_dict-like dict of tensors).")
        return 7


if __name__ == "__main__":
    raise SystemExit(main())
