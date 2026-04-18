"""
config.py
---------
Single source of truth for all training hyperparameters.
Tuned for a local CUDA GPU setup.

Edit this file instead of passing CLI flags for cleaner experiment tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    # ── Dataset ───────────────────────────────────────────────────────────────
    num_frames: int = 24          # Frames sampled per video (more = better temporal coverage)
    img_size: int = 224           # Frame size. 224 matches EfficientNet pretrain resolution
    num_workers: int = 4          # DataLoader workers. Match to your CPU core count
    pin_memory: bool = True       # Keep True for CUDA — speeds up CPU→GPU transfer

    # Target metrics (eye contact + facial expression focused)
    target_cols: List[str] = field(default_factory=lambda: [
        "extraversion",           # Social expressiveness, visible in face/body
        "confidence",             # Gaze steadiness, posture, expression
        "engagement",             # Animated expressions, eye contact, energy
        "professional_appearance",# Composure, neutral baseline expression
        "overall_performance",    # Combined interview score
    ])

    # ── Model ─────────────────────────────────────────────────────────────────
    fusion_dim: int = 512         # Fused embedding dimension
    num_heads: int = 8            # Transformer attention heads (must divide fusion_dim)
    num_transformer_layers: int = 3  # Depth of temporal transformer
    dropout: float = 0.2

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 25
    batch_size: int = 16          # Good for 8GB VRAM. Reduce to 8 if OOM.
    accumulation_steps: int = 2   # Effective batch = batch_size × accumulation_steps

    # Learning rates
    lr_head: float = 3e-4         # LR for new layers (head, fusion, transformer)
    lr_backbone: float = 3e-5     # LR for EfficientNet after unfreezing (10× lower)
    weight_decay: float = 1e-4
    warmup_epochs: int = 2        # Linear LR warmup before cosine decay

    # Two-phase training
    freeze_epochs: int = 5        # Epochs to keep backbone frozen before fine-tuning

    # ── Regularization ────────────────────────────────────────────────────────
    label_smoothing: float = 0.0  # Not used for regression; kept as placeholder
    mixup_alpha: float = 0.0      # Set >0 to enable MixUp augmentation

    # ── Early stopping ────────────────────────────────────────────────────────
    patience: int = 7             # Stop if val_loss doesn't improve for this many epochs
    min_delta: float = 1e-4       # Minimum improvement to count

    # ── CUDA / Performance ────────────────────────────────────────────────────
    use_amp: bool = True          # Mixed precision (fp16). Big speedup on CUDA.
    compile_model: bool = False   # torch.compile() — faster but adds startup time
                                  # Set True if you have PyTorch 2.0+ and patience

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "eye_contact_expression_v1"

    # ── Debug ─────────────────────────────────────────────────────────────────
    debug: bool = False           # If True, uses only 10 samples for quick validation
    debug_samples: int = 10


# Default config instance — import this in other files
CFG = TrainConfig()
