"""
model.py
--------
InterviewAnalysisNet: EfficientNet-B0 backbone + MediaPipe face geometry
fusion + Temporal Transformer for interview video analysis.

Architecture summary:
  frames [B,T,3,H,W] ──► EfficientNet-B0 ──► 1280-dim/frame
                                                    │
  face geometry [B,T,1440] ──► FaceGeometryMLP ──► 256-dim/frame
                                                    │
                            concat + FusionMLP ──► fusion_dim/frame  [B,T,D]
                                                    │
                            Temporal Transformer ──► CLS token [B,D]
                                                    │
                            Regression head ──► [B, num_targets]

Training phases:
  Phase 1 (epochs 0-freeze_epochs): backbone frozen, only new layers train
  Phase 2 (epoch freeze_epochs+):   full model fine-tunes with differential LRs
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from backend.FacialExpressionsAndEyeContact.data_loader import FaceFeatureExtractor


class SpatialEncoder(nn.Module):
    """EfficientNet-B0 pretrained backbone — extracts 1280-dim feature per frame."""

    OUT_DIM = 1280

    def __init__(self, freeze: bool = True):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features   # [B, 1280, H', W']
        self.pool = backbone.avgpool        # → [B, 1280, 1, 1]

        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*T, 3, H, W] → [B*T, 1280]"""
        return self.pool(self.features(x)).flatten(1)


class FaceGeometryEncoder(nn.Module):
    """
    MLP that compresses 1440-dim MediaPipe landmark + derived features
    into a compact 256-dim embedding for fusion with CNN features.
    """

    def __init__(self, in_dim: int = FaceFeatureExtractor.FEATURE_DIM, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*T, in_dim] → [B*T, out_dim]"""
        return self.net(x)


class TemporalTransformer(nn.Module):
    """
    Standard Transformer encoder over a sequence of T frame embeddings.
    A learnable CLS token aggregates the full sequence into one vector.
    Supports up to 64 frames via positional embedding.
    """

    MAX_FRAMES = 64

    def __init__(self, d_model: int, num_heads: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.MAX_FRAMES + 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, D]"""
        B, T, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)       # [B, 1, D]
        x = torch.cat([cls, x], dim=1)               # [B, T+1, D]
        x = x + self.pos_embed[:, :T + 1]
        x = self.transformer(x)
        return x[:, 0]                                # CLS output [B, D]


class InterviewAnalysisNet(nn.Module):
    """
    Full model for predicting eye contact quality and facial expression
    scores from an interview video clip.

    Args:
        num_targets   : Number of continuous scores to predict
        fusion_dim    : Per-frame embedding dimension after CNN + face fusion
        num_heads     : Transformer attention heads (must divide fusion_dim)
        num_layers    : Transformer encoder depth
        dropout       : Dropout rate (applied in transformer + head)
        freeze_backbone: Whether to freeze EfficientNet at init (Phase 1)
    """

    def __init__(
        self,
        num_targets: int = 5,
        fusion_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.spatial_encoder = SpatialEncoder(freeze=freeze_backbone)
        cnn_dim = SpatialEncoder.OUT_DIM  # 1280

        self.face_encoder = FaceGeometryEncoder(
            in_dim=FaceFeatureExtractor.FEATURE_DIM,
            out_dim=256,
        )
        face_dim = self.face_encoder.out_dim  # 256

        # Project concatenated [CNN | face] → fusion_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(cnn_dim + face_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.temporal = TemporalTransformer(
            d_model=fusion_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, num_targets),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize new (non-pretrained) layers with sensible defaults."""
        for m in [self.face_encoder, self.fusion_proj, self.head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, frames: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames  : [B, T, 3, H, W]   – normalized video frames
            features: [B, T, face_feat] – MediaPipe landmark features

        Returns:
            [B, num_targets] – predicted continuous scores
        """
        B, T, C, H, W = frames.shape

        # ── Per-frame spatial encoding (all frames in one fwd pass) ──────────
        cnn_out = self.spatial_encoder(frames.view(B * T, C, H, W))    # [B*T, 1280]
        face_out = self.face_encoder(features.view(B * T, -1))          # [B*T, 256]

        # ── Fusion ────────────────────────────────────────────────────────────
        fused = self.fusion_proj(torch.cat([cnn_out, face_out], dim=1)) # [B*T, D]
        fused = fused.view(B, T, -1)                                    # [B, T, D]

        # ── Temporal modeling ─────────────────────────────────────────────────
        video_repr = self.temporal(fused)                               # [B, D]

        # ── Regression ────────────────────────────────────────────────────────
        return self.head(video_repr)                                    # [B, num_targets]


def build_model(num_targets: int, cfg=None, freeze_backbone: bool = True) -> InterviewAnalysisNet:
    """
    Factory function. Pass a TrainConfig to use config-driven hyperparameters,
    or pass num_targets + freeze_backbone for a quick default model.
    """
    if cfg is not None:
        model = InterviewAnalysisNet(
            num_targets=num_targets,
            fusion_dim=cfg.fusion_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_transformer_layers,
            dropout=cfg.dropout,
            freeze_backbone=True,
        )
    else:
        model = InterviewAnalysisNet(
            num_targets=num_targets,
            freeze_backbone=freeze_backbone,
        )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"InterviewAnalysisNet: {total:,} params total | {trainable:,} trainable")
    return model


if __name__ == "__main__":
    # Sanity check
    from backend.FacialExpressionsAndEyeContact.data_loader import FaceFeatureExtractor
    model = build_model(num_targets=5)
    B, T = 2, 24
    frames   = torch.randn(B, T, 3, 224, 224)
    features = torch.randn(B, T, FaceFeatureExtractor.FEATURE_DIM)
    out = model(frames, features)
    print(f"Output shape: {out.shape}")   # expected: [2, 5]
    assert out.shape == (B, 5), f"Shape mismatch: {out.shape}"
    print("Model sanity check passed.")
