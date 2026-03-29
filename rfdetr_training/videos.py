from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
import cv2

def extract_frames(
    video_paths: List[Path],
    out_dir: Path,
    total_frames: int,
    verbose: bool = True,
) -> int:
    """
    Extract a total of `total_frames` from `video_paths`, distributed proportionally by duration.
    Frames are sampled uniformly from each video.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Get info about all videos
    video_infos = []
    total_available_frames = 0
    
    for v_path in video_paths:
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            if verbose:
                print(f"Warning: Could not open video {v_path}")
            continue
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            # Try to count frames manually if metadata is missing/wrong
            count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
            frame_count = count
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        video_infos.append({
            "path": v_path,
            "available": frame_count,
        })
        total_available_frames += frame_count
        cap.release()

    if not video_infos:
        raise RuntimeError("No valid videos found.")

    if total_available_frames == 0:
        raise RuntimeError("No frames available in the provided videos.")

    # 2. Calculate how many frames to take from each video
    # Proportional distribution
    frames_to_extract = []
    current_total = 0
    for info in video_infos:
        count = int((info["available"] / total_available_frames) * total_frames)
        # Ensure we don't exceed available frames
        count = min(count, info["available"])
        frames_to_extract.append(count)
        current_total += count
        
    # Adjust if we are slightly off due to rounding
    diff = total_frames - current_total
    if diff > 0:
        # Add remaining to the video with most available frames that still has room
        for _ in range(diff):
            # Find video with most remaining capacity (proportional)
            best_idx = -1
            max_rem = -1.0
            for i, info in enumerate(video_infos):
                if frames_to_extract[i] < info["available"]:
                    rem = info["available"] - frames_to_extract[i]
                    if rem > max_rem:
                        max_rem = rem
                        best_idx = i
            if best_idx != -1:
                frames_to_extract[best_idx] += 1
            else:
                break
    elif diff < 0:
        # Remove from videos
        for _ in range(abs(diff)):
            best_idx = -1
            max_val = -1
            for i in range(len(frames_to_extract)):
                if frames_to_extract[i] > 0:
                    if frames_to_extract[i] > max_val:
                        max_val = frames_to_extract[i]
                        best_idx = i
            if best_idx != -1:
                frames_to_extract[best_idx] -= 1
            else:
                break

    # 3. Extract frames
    extracted_count = 0
    for info, count in zip(video_infos, frames_to_extract):
        if count <= 0:
            continue
            
        if verbose:
            print(f"Extracting {count} frames from {info['path'].name}...")
            
        cap = cv2.VideoCapture(str(info['path']))
        avail = info["available"]
        
        # Uniform sampling: step = avail / count
        # We want to pick frames at indices: floor(i * step) for i in 0..count-1
        step = avail / count
        
        for i in range(count):
            frame_idx = int(i * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # Try next frame if this one fails
                continue
            
            out_name = f"{info['path'].stem}_frame_{frame_idx:06d}.jpg"
            out_path = out_dir / out_name
            
            # Avoid overwriting if same filename exists from another video (unlikely with stem)
            counter = 1
            while out_path.exists():
                out_path = out_dir / f"{info['path'].stem}_frame_{frame_idx:06d}_{counter}.jpg"
                counter += 1
                
            cv2.imwrite(str(out_path), frame)
            extracted_count += 1
            
        cap.release()
        
    return extracted_count
