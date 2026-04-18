# Tone modeling: transcript embeddings, PKL export, and transformer regressor

This note summarizes how we represent **per-participant** segment data, why we use a **`.pkl`** file in addition to CSV, how **`NN.ipynb`** builds the model (multimodal text + prosody), training hyperparameters (including **weight decay**), **example results**, and a **current bottleneck** (overfitting).

---

## 1. Why we need `deep-prep-ai-audio-embeddings.pkl`

### What goes wrong with CSV alone

In `Audio_Embedding.ipynb`, the table is exported with:

```text
table.to_csv("deep-prep-ai-audio-embeddings.csv", index=False)
```

When a column holds **NumPy arrays** (or nested lists of floats), pandas often **stringifies** those values for CSV. Long arrays are not written in full: the text representation is **truncated** with an ellipsis (`...`) in the middle, e.g. a row might look like `"[[0.12 0.03 ... 0.41] ...]"` instead of the full `(n_segments × embed_dim)` matrix.

**You cannot recover the true floating-point vectors from that string.** Any downstream step that reads CSV and expects real embeddings would be using **corrupted or incomplete** data.

### What the PKL file does

The same notebook saves a **binary pickle** payload so nothing is truncated:

- Arrays stay as **NumPy arrays** (or lists of arrays) with full precision.
- The export comment in the notebook states explicitly that CSV can truncate with `"..."`, so we **also** save the real arrays to pickle.

The payload includes (among others):

| Key                     | Role                                                                 |
| ----------------------- | -------------------------------------------------------------------- |
| `participant_id`        | Aligns rows across modalities                                      |
| `transcript_segments`   | Text segments (list)                                               |
| `prosody_features`      | Segment-aligned prosody `(n_segments, 13)`                         |
| `transcript_embeddings` | **Per-participant** `(n_segments, embed_dim)` text embeddings       |
| Label columns           | e.g. `interview_score`, `overall_personality`, traits, etc.        |

`NN.ipynb` loads **`transcript_embeddings`**, **`prosody_features`**, and **label columns** from this PKL so training uses **lossless** tensors, not CSV strings.

---

## 2. How the model is built (`MultimodalTransformerRegressor` in `NN.ipynb`)

Each participant has **two aligned segment modalities**: **text** (e.g. 384-d sentence-transformer embeddings per segment) and **prosody** (13-d features per segment). They share the same **`key_padding_mask`** (padded to **`MAX_SEQ_LEN = 64`**).

1. **`ModalitySeqEncoder` (×2)** — separate transformer stacks for text and for prosody: each maps segments to **`d_model = 128`**, uses a **CLS** token + **positional** embeddings, **`nn.TransformerEncoder`**, and returns the **CLS vector**.
2. **Fusion** — default **`FUSION = "concat"`**: concatenate the two CLS vectors, **`Linear(256 → 128)`**, then a **multi-target** head (`num_targets` outputs). Optional **`FUSION = "attention"`** uses a short 2-token self-attention block then mean-pool.
3. **Head** — `LayerNorm → Linear → GELU → Dropout → Linear` to all regression targets (mean **MSE** over targets).

---

## 3. Training setup

Rough picture of the training loop in `NN.ipynb`:

| Idea                                     | Purpose                                                                                                                                                                                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AdamW**                                | Adaptive optimizer with decoupled **weight decay**.                                                                                                                                                                                                |
| **Fixed learning rate**                  | e.g. **`LR = 1e-4`** for the whole run (no `ReduceLROnPlateau` in recent experiments — easier to read whether loss trends are from LR decay vs generalization).                                                                                    |
| **Gradient clipping** (`max_norm = 1.0`) | Limits spike updates from individual batches.                                                                                                                                                                                                     |
| **Early stopping**                       | After **`MIN_EPOCHS_BEFORE_STOP`**, count **`PATIENCE`** consecutive epochs without validation improvement (above **`MIN_DELTA`**); restore **best validation** weights on CPU.                                                                    |

Regularization knobs (see notebook): **`WEIGHT_DECAY`**, **`DROPOUT`**, **`FUSION`**.

---

## 4. Weight decay (explicit answer)

We use **AdamW** with:

```text
WEIGHT_DECAY = 1e-3   # AdamW; see NN.ipynb (sweep 0, 1e-5, 1e-4, 1e-3)
DROPOUT = 0.25        # encoder(s) + head
```

So the current default **L2-style weight decay** is **`1e-3`**, with **dropout `0.25`**. You can sweep **`1e-4`**, **`1e-5`**, **`0`** if train keeps dropping while validation rises.

---

## 5. Example training run results (logged from `NN.ipynb`)

One representative run (multimodal model, **fixed** `LR = 1e-4`, AdamW + weight decay + dropout as above, early stopping enabled):

| Epoch | Train MSE | Val MSE   | LR        |
| ----- | --------- | --------- | ---------- |
| 10    | 0.9701    | 1.0374    | 1.00e-04 (fixed) |
| 15    | 0.9150    | 1.0747    | 1.00e-04 (fixed) |
| 20    | 0.8511    | 1.1355    | 1.00e-04 (fixed) |

**Early stopping:** epoch **22** (per run configuration).  
**Best validation MSE:** **1.0265** at **epoch 8** (checkpoint restored at the end of training).

*Numbers are from a single train/val split; re-running the notebook can change them slightly.*

---

## 6. Current bottleneck: overfitting on the training split

Right now the main limitation we see in logs is **overfitting**: **training MSE keeps improving** (e.g. from ~0.97 toward ~0.85 over epochs 10–20) while **validation MSE gets worse** (e.g. from ~1.04 toward ~1.14). That **widening train–val gap** means the network is **memorizing patterns specific to the training subset** (including noise) instead of learning a rule that **generalizes** to held-out participants. Even with **weight decay** and **dropout**, capacity (two encoders + fusion + nine targets) can still exceed what the split stably supports.

Directions that often help (see course-style tuning): **stronger or weaker regularization**, **smaller encoders**, **k-fold or different val splits**, **per-target normalization**, or **simpler baselines** to check how much signal is in text vs prosody.

---

## 7. File map

| File                                | Role                                                                                |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| `Audio_Embedding.ipynb`             | Builds `table`, writes CSV + **`deep-prep-ai-audio-embeddings.pkl`**.               |
| `deep-prep-ai-audio-embeddings.csv` | Human-readable / spreadsheet-friendly; **not** reliable for full embedding tensors. |
| `deep-prep-ai-audio-embeddings.pkl` | **Authoritative** store for arrays and training.                                    |
| `NN.ipynb`                          | Loads PKL, pads sequences, trains **`MultimodalTransformerRegressor`**.             |

---

_Last aligned with `NN.ipynb` and `Audio_Embedding.ipynb` in this repo; re-open those notebooks if hyperparameters drift._
