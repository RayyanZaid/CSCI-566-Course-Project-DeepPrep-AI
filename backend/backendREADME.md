# DeepPrep AI — Backend

Neural network pipeline for eye contact and facial expression analysis in interview videos.

---

## Files

| File | Purpose |
|------|---------|
| `explore_dataset.py` | **Start here.** Inspect RecruitView dataset structure and discover exact column names. |
| `data_loader.py` | HuggingFace dataset wrapper + MediaPipe face feature extractor |
| `model.py` | `InterviewAnalysisNet` — CNN + Transformer architecture |
| `train.py` | Full training loop with checkpointing |
| `inference.py` | Run trained model on a new MP4 video |

---

## Quick Start

### Step 1 — Install new dependency
```bash
pip install mediapipe
pip freeze > requirements.txt
```

### Step 2 — Explore the dataset first
```bash
python backend/explore_dataset.py
```
This will print all column names and score distributions. **Important:** if the dataset uses different column names than expected, update `VISUAL_TARGETS` in `data_loader.py`.

### Step 3 — Smoke test training
```bash
# Quick 10-sample test to verify everything runs end-to-end
python backend/train.py --debug --epochs 2 --bs 2
```

### Step 4 — Full training
```bash
# With GPU (recommended)
python backend/train.py --epochs 20 --bs 8

# CPU only (slow but works)
python backend/train.py --epochs 10 --bs 2 --workers 0

# Choose specific targets
python backend/train.py --targets confidence engagement professional_appearance
```

### Step 5 — Run inference on a new video
```bash
python backend/inference.py --video my_interview.mp4
python backend/inference.py --video my_interview.mp4 --report  # saves JSON
```

---

## Architecture Overview

```
Video
  │
  ▼
[Sample T=16 frames uniformly]
  │
  ├──► EfficientNet-B0 backbone ──► 1280-dim CNN features
  │         (pretrained, fine-tuned in phase 2)
  │
  ├──► MediaPipe FaceMesh ──► 1410-dim face geometry features
  │         (468 landmarks × 3 + 6 derived features)
  │         (eye openness, gaze proxy, mouth ratio)
  │
  └──► Fusion MLP → 512-dim per-frame embedding
            │
            ▼
       Transformer Encoder (2 layers, 4 heads)
            │ CLS token
            ▼
       Regression Head → N target scores
```

**Two-phase training:**
- **Phase 1** (epochs 0–5): Backbone frozen, only fusion + transformer + head train
- **Phase 2** (epochs 5+): Full fine-tuning with lower backbone LR

---

## Target Metrics

The RecruitView dataset has 12 continuous target scores (normalized, centered ~0), rated by clinical psychologists via pairwise comparisons:

**Big Five Personality:**
- `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`

**Interview Performance (most visually relevant):**
- `confidence`, `engagement`, `communication`, `clarity`, `professional_appearance`, `overall_performance`

**Overall:** `overall_personality`

By default we train on the 5 most visually predictable ones:
`extraversion`, `confidence`, `engagement`, `professional_appearance`, `overall_performance`

---

## Tips for Better Results

1. **Run `explore_dataset.py` first** — verify exact column names before training
2. **Use a GPU** — each training epoch is ~30 min on CPU, ~2 min on a good GPU
3. **Google Colab** — free T4 GPU is sufficient; just `pip install mediapipe` at the top
4. **Increase `num_frames`** (e.g., 32) for better temporal coverage if GPU memory allows
5. **Check `face_detection_rate`** in inference output — if it's below 50%, the video quality or angle may be limiting predictions
