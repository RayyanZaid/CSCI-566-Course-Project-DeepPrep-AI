import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


LABEL_COLS = [
    "interview_score",
    "overall_personality",
    "answer_score",
    "speaking_skills",
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
    "confidence_score",
]

def to_float32_label(col_values):
    return pd.to_numeric(pd.Series(col_values), errors="coerce").to_numpy(dtype=np.float32)



def parse_embedding_entry(entry):
    if isinstance(entry, np.ndarray):
        arr = entry.astype(np.float32, copy=False)
    elif isinstance(entry, (list, tuple)):
        arr = np.array(entry, dtype=np.float32)
    elif isinstance(entry, str):
        cleaned = (
            entry.replace("\n", " ")
            .replace("[", " ")
            .replace("]", " ")
            .replace(",", " ")
        )
        flat = np.fromstring(cleaned, sep=" ", dtype=np.float32)
        if flat.size == 0:
            return None
        arr = flat
    else:
        return None

    if arr.ndim == 0:
        return None
    if arr.ndim > 2:
        return None
    return arr




# ── 2) Dataset (text + prosody modalities) ─────────────────────────────────────
class MultimodalSeqDataset(Dataset):
    def __init__(self, X_text, X_prosody, key_padding_mask, y):
        self.X_text = torch.tensor(X_text, dtype=torch.float32)
        self.X_prosody = torch.tensor(X_prosody, dtype=torch.float32)
        self.key_padding_mask = torch.tensor(key_padding_mask, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_text[idx],
            self.X_prosody[idx],
            self.key_padding_mask[idx],
            self.y[idx],
        )

# ── 3) Model: separate encoders (text vs prosody) + fusion + head ───────────
class ModalitySeqEncoder(nn.Module):
    """Transformer over one modality's segment features -> CLS vector (d_model,)."""

    def __init__(
        self,
        input_dim,
        max_seq_len,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, key_padding_mask):
        b, seq_len, _ = x.shape
        x = self.input_proj(x)
        cls = self.cls_token.expand(b, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : seq_len + 1, :]
        cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=full_mask)
        return x[:, 0, :]


class TwoTokenFusion(nn.Module):
    """Self-attention over [text_vec, audio_vec] then mean-pool."""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, text_vec, audio_vec):
        x = torch.stack([text_vec, audio_vec], dim=1)
        x = self.encoder(x)
        return x.mean(dim=1)


class MultimodalTransformerRegressor(nn.Module):
    def __init__(
        self,
        text_dim,
        prosody_dim,
        max_seq_len,
        num_targets,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        fusion="concat",
    ):
        super().__init__()
        self.fusion = fusion
        self.text_encoder = ModalitySeqEncoder(
            text_dim, max_seq_len, d_model, nhead, num_layers, dropout
        )
        self.audio_encoder = ModalitySeqEncoder(
            prosody_dim, max_seq_len, d_model, nhead, num_layers, dropout
        )
        if fusion == "concat":
            self.fuse_proj = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.two_token_fusion = None
        elif fusion == "attention":
            self.two_token_fusion = TwoTokenFusion(d_model, nhead, dropout)
            self.fuse_proj = None
        else:
            raise ValueError('fusion must be "concat" or "attention"')

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_targets),
        )

    def forward(self, x_text, key_padding_mask, x_prosody):
        text_repr = self.text_encoder(x_text, key_padding_mask)
        audio_repr = self.audio_encoder(x_prosody, key_padding_mask)
        if self.fusion == "concat":
            fused = self.fuse_proj(torch.cat([text_repr, audio_repr], dim=-1))
        else:
            fused = self.two_token_fusion(text_repr, audio_repr)
        return self.head(fused)


