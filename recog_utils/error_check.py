import numpy as np
import re

def extract_frames(filename: str, num_frames=8):
    # Extract the number before ".npy" using regex
    match = re.search(r'_(\d+)\.npy$', filename)
    if not match:
        raise ValueError("Filename format is incorrect, expected a number before .npy")

    frame_idx = int(match.group(1))  # Convert to integer

    # Clean filename by removing the number before ".npy"
    cleaned_filename = re.sub(r'_\d+\.npy$', '.npy', filename)
    x = np.load("/nfs/wattrel/data/md0/datasets/action_seg_datasets/breakfast/hiervl_split5/combined_feat/P24_cam01_P24_cereals.npy")

    # Ensure we don't go out of bounds
    T = x.shape[1]
    start_idx = max(0, min(frame_idx, T - num_frames))  # Prevent overflow

    frames = x[:, start_idx:start_idx + num_frames]

    return frames, cleaned_filename

