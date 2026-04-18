"""
explore_dataset.py
------------------
One-shot script to understand the RecruitView dataset structure
before committing to a full training run.

Run: python explore_dataset.py
"""

from datasets import load_dataset
import numpy as np

print("Loading dataset (first 5 samples only for speed)...")
dataset = load_dataset("AI4A-lab/RecruitView", split="train")

print(f"\nTotal samples: {len(dataset)}")
print(f"\nColumn names:")
sample = dataset[0]
for k, v in sample.items():
    if k == "video":
        print(f"  {k}: <Video object, {v._num_frames} frames>")
    elif isinstance(v, str) and len(v) > 80:
        print(f"  {k}: str[{len(v)}] = '{v[:60]}...'")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

# Identify numeric (target) columns
print("\n── Numeric columns (potential target metrics) ──")
numeric_cols = []
for k, v in sample.items():
    if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in {"id", "user_no", "question_id"}:
        numeric_cols.append(k)
        print(f"  {k}: {v}")

# Collect stats across 50 samples
print(f"\n── Score distribution (first 50 samples) ──")
if numeric_cols:
    scores = {col: [] for col in numeric_cols}
    for i in range(min(50, len(dataset))):
        s = dataset[i]
        for col in numeric_cols:
            if s.get(col) is not None:
                scores[col].append(float(s[col]))

    for col, vals in scores.items():
        if vals:
            arr = np.array(vals)
            print(f"  {col:30s}: mean={arr.mean():.3f}  std={arr.std():.3f}  "
                  f"min={arr.min():.3f}  max={arr.max():.3f}")

print("\n── Video info (first sample) ──")
vid = sample["video"]
print(f"  Frames : {vid._num_frames}")
frame0 = vid[0].data
print(f"  Shape  : {frame0.shape}  (C x H x W)")
print(f"  dtype  : {frame0.dtype}")

print("\nExploration complete. Update TARGET_COLUMNS in data_loader.py if needed.")
