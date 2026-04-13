# Tone modeling: transcript embeddings, PKL export, and transformer regressor

This note summarizes how we represent **per-participant transcript segment embeddings**, why we use a **`.pkl`** file in addition to CSV, how the **transformer-style regressor** in `NN.ipynb` is built, and which **training hyperparameters** (including **weight decay**) we use.

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

| Key                     | Role                                                                        |
| ----------------------- | --------------------------------------------------------------------------- |
| `participant_id`        | Aligns rows across modalities                                               |
| `overall_personality`   | Regression target                                                           |
| `transcript_segments`   | Text segments (list)                                                        |
| `prosody_features`      | Segment-aligned prosody (list of arrays)                                    |
| `transcript_embeddings` | **Per-participant** `(n_segments, embed_dim)` embeddings used by `NN.ipynb` |

`NN.ipynb` loads **`transcript_embeddings`** and **`overall_personality`** from this PKL file so training uses **lossless** segment tensors, not CSV strings.

---

## 2. How the transformer architecture was built (`TransformerSeqRegressor`)

The model treats each participant as a **sequence of segment embeddings** (same segment structure as prosody), not a single flat vector.

### Inputs and padding

- Each participant has a variable number of segments; we cap length at **`MAX_SEQ_LEN = 64`**.
- We build **`X`** with shape `(N, MAX_SEQ_LEN, embed_dim)` and a **`key_padding_mask`**: `True` where there is no real segment (padding), `False` on real tokens.
- Embeddings are projected from **`embed_dim`** (e.g. 384 from the encoder) to **`d_model = 128`** via `nn.Linear`.

### Encoder-style stack (BERT-like pattern, simplified)

1. **`nn.Linear(input_dim → d_model)`** — project each segment.
2. **Learnable `[CLS]` token** — prepended as an extra “summary” position (not padded).
3. **Learnable positional embeddings** — added for positions `0 … seq_len` (CLS + segments).
4. **`nn.TransformerEncoder`** — `batch_first=True`, with:
   - **`d_model = 128`**, **`nhead = 4`**, **`num_layers = 2`**
   - **Feedforward dim** `4 × d_model`, **dropout** `0.1`
5. **Padding mask** — `src_key_padding_mask` masks padded **segment** positions; the CLS position is **never** masked so it always receives attention.

### Regression head

The vector at the **CLS** position (`x[:, 0, :]`) is passed through:

`LayerNorm → Linear(128 → 64) → GELU → Dropout → Linear(64 → 1)`

Output is a **scalar per participant** (MSE against `overall_personality`).

---

## 3. Training setup and “what changed”

Rough evolution of the training loop in `NN.ipynb`:

| Idea                                     | Purpose                                                                                                                                                                                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AdamW**                                | Standard adaptive optimizer with decoupled weight decay.                                                                                                                                                                                          |
| **Base learning rate**                   | Tuned over time (e.g. from `1e-3` down toward **`1e-4`** or lower); very small LRs (e.g. **`1e-6`**) may be used when exploring stability—**check the `LR = …` line in `NN.ipynb` for the value you are actually running.**                       |
| **`ReduceLROnPlateau`**                  | Reduces LR when validation loss stops improving (mode `min`, factor `0.5`, patience `2`, `min_lr` floor).                                                                                                                                         |
| **Gradient clipping** (`max_norm = 1.0`) | Limits spike updates from individual batches.                                                                                                                                                                                                     |
| **Early stopping**                       | Stop if validation does not improve by at least **`MIN_DELTA`** for **`PATIENCE`** consecutive epochs **after** **`MIN_EPOCHS_BEFORE_STOP`**, to avoid stopping at epoch ~6 when epoch-1 validation is noisy. Best weights are restored from CPU. |

This matches the spirit of a **coarse hyperparameter sweep**: try a few **learning rates** and **weight decay** values, train for a **small number of epochs** first to see loss behavior, then run longer with the stable region you find.

---

## 4. Weight decay (explicit answer)

We use **AdamW** with:

```text
weight_decay = 1e-4
```

So our current **L2-style weight decay coefficient is `1e-4`**.

Course-style grids often try values such as **`1e-4`**, **`1e-5`**, and **`0`**; we are on the **`1e-4`** branch unless you change that line in `NN.ipynb`. Sweeps over `1e-5` or `0` are reasonable next steps if you want to compare generalization.

---

## 5. File map

| File                                | Role                                                                                |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| `Audio_Embedding.ipynb`             | Builds `table`, writes CSV + **`deep-prep-ai-audio-embeddings.pkl`**.               |
| `deep-prep-ai-audio-embeddings.csv` | Human-readable / spreadsheet-friendly; **not** reliable for full embedding tensors. |
| `deep-prep-ai-audio-embeddings.pkl` | **Authoritative** store for arrays and training.                                    |
| `NN.ipynb`                          | Loads PKL, pads sequences, trains **`TransformerSeqRegressor`**.                    |

---

_Last aligned with `NN.ipynb` and `Audio_Embedding.ipynb` in this repo; re-open those notebooks if hyperparameters drift._
