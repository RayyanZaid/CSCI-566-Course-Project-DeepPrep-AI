"""
train.py
--------
CUDA-optimized training pipeline for InterviewAnalysisNet.

Key features:
  - cudnn.benchmark=True for fixed-size input speedup
  - torch.compile() support (optional, PyTorch 2.0+)
  - Mixed precision (fp16) via torch.cuda.amp
  - Gradient accumulation (effective batch scaling without extra VRAM)
  - Linear warmup → cosine decay LR schedule
  - Two-phase training: frozen backbone → full fine-tune with differential LRs
  - Early stopping with configurable patience
  - Per-metric val loss breakdown (see which targets the model learns best)
  - Clean checkpoint saving: best model + periodic snapshots

Usage:
    python train.py                          # uses config.py defaults
    python train.py --debug                  # 10-sample smoke test
    python train.py --epochs 30 --bs 8       # override epochs and batch size
    python train.py --batch-size 4           # --bs and --batch-size are equivalent
    python train.py --workers 4              # override DataLoader workers
    python train.py --targets confidence engagement  # override target metrics
    python train.py --compile                # enable torch.compile (slower start, faster run)
    python train.py --debug --epochs 5       # 5-epoch debug run (CLI epochs wins over debug default)
"""

import argparse
import time
import math
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from config import CFG, TrainConfig
from data_loader import get_dataloaders
from model import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler: linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr_lambda(warmup_epochs: int, total_epochs: int):
    """Returns a LambdaLR multiplier function."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear warmup
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress)) # cosine decay
    return lr_lambda


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, target_cols, num_frames, path):
    # Unwrap torch.compile wrapper if present
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "epoch": epoch,
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss": val_loss,
        "target_cols": target_cols,
        "num_frames": num_frames,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["val_loss"], ckpt.get("target_cols", [])


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation step
# ─────────────────────────────────────────────────────────────────────────────

# Huber loss is more robust than MSE for the roughly centered, slightly skewed
# score distributions in RecruitView
criterion = nn.HuberLoss(delta=1.0, reduction="none")  # keep per-target for logging


def run_epoch(model, loader, optimizer, scaler, device, cfg, train=True, accumulation_steps=1):
    """
    Run one full epoch.

    Returns:
        mean_loss : float — average total loss across all batches
        per_target: list[float] — average loss per target metric
    """
    model.train(train)
    total_loss = 0.0
    per_target_loss = None
    n_batches = 0

    if train:
        optimizer.zero_grad()

    for step, batch in enumerate(loader):
        frames   = batch["frames"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        labels   = batch["labels"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast(device_type=device.type, enabled=cfg.use_amp and device.type == "cuda"):
                preds = model(frames, features)                    # [B, num_targets]
                loss_per_target = criterion(preds, labels)         # [B, num_targets]
                loss = loss_per_target.mean()                      # scalar

                if train and accumulation_steps > 1:
                    loss = loss / accumulation_steps

        if train:
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Accumulate stats (undo accumulation scaling for logging)
        raw_loss = loss.item() * (accumulation_steps if train else 1)
        total_loss += raw_loss

        per_t = loss_per_target.mean(dim=0).detach().cpu()  # [num_targets]
        if per_target_loss is None:
            per_target_loss = per_t
        else:
            per_target_loss += per_t
        n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    per_target_avg = (per_target_loss / max(n_batches, 1)).tolist() if per_target_loss is not None else []
    return mean_loss, per_target_avg


# ─────────────────────────────────────────────────────────────────────────────
# Backbone phase management
# ─────────────────────────────────────────────────────────────────────────────

def make_optimizer_phase1(model, cfg):
    """Phase 1: backbone frozen — only train head, fusion, transformer."""
    params = filter(lambda p: p.requires_grad, model.parameters())
    return AdamW(params, lr=cfg.lr_head, weight_decay=cfg.weight_decay)


def make_optimizer_phase2(model, cfg):
    """Phase 2: full fine-tune with differential LRs (backbone gets 10× lower LR)."""
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    return AdamW([
        {"params": raw_model.spatial_encoder.parameters(), "lr": cfg.lr_backbone},
        {"params": raw_model.face_encoder.parameters(),    "lr": cfg.lr_head},
        {"params": raw_model.fusion_proj.parameters(),     "lr": cfg.lr_head},
        {"params": raw_model.temporal.parameters(),        "lr": cfg.lr_head},
        {"params": raw_model.head.parameters(),            "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)


def unfreeze_backbone(model):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    for p in raw_model.spatial_encoder.features.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"  Backbone unfrozen. Trainable params: {n:,}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        cudnn.benchmark = True   # fastest conv algorithm for fixed input shapes
        cudnn.deterministic = False
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("Apple MPS")
    else:
        dev = torch.device("cpu")
        print("CPU (slow — consider Google Colab for GPU)")
    return dev


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--debug",   action="store_true", help="10-sample smoke test")
    p.add_argument("--epochs",  type=int, default=None, help="Override number of training epochs")
    p.add_argument("--bs", "--batch-size", dest="batch_size", type=int, default=None,
                   help="Override batch size")
    p.add_argument("--workers", type=int, default=None, help="Override DataLoader workers")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Override target columns, e.g. --targets confidence engagement")
    p.add_argument("--resume",  type=str, default=None, help="Checkpoint path to resume from")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile() for extra speed (PyTorch 2.0+ only)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = copy.deepcopy(CFG)  # keep global defaults immutable per run

    # Apply debug defaults first, then let explicit CLI flags override them.
    # This means --debug --epochs 5 gives a 5-epoch debug run, not a forced 2-epoch one.
    if args.debug:
        cfg.debug = True
        cfg.epochs = 2
        cfg.batch_size = 2

    # CLI overrides always win, even over --debug defaults
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.workers is not None:
        cfg.num_workers = args.workers
    if args.targets:
        cfg.target_cols = args.targets

    device = get_device()

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, target_cols = get_dataloaders(cfg)
    num_targets = len(target_cols)
    print(f"Targets ({num_targets}): {target_cols}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_targets=num_targets, cfg=cfg, freeze_backbone=True).to(device)

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # ── Optimizer + scheduler (Phase 1) ──────────────────────────────────────
    optimizer = make_optimizer_phase1(model, cfg)
    # Guard: if freeze_epochs >= total epochs (e.g. short debug runs),
    # phase 2 will never trigger — build scheduler for the full run normally.
    lr_lambda = get_lr_lambda(min(cfg.warmup_epochs, cfg.epochs - 1), cfg.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    start_epoch = 0
    best_val_loss = float("inf")
    early_stop = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt = ckpt_dir / f"{cfg.experiment_name}_best.pt"

    if args.resume:
        start_epoch, best_val_loss, _ = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        print(f"Resumed from epoch {start_epoch} | best val: {best_val_loss:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  Training: {cfg.experiment_name}")
    print(f"  Epochs: {cfg.epochs} | BS: {cfg.batch_size} "
          f"(×{cfg.accumulation_steps} accum = {cfg.batch_size * cfg.accumulation_steps} effective)")
    print(f"  AMP: {cfg.use_amp} | cudnn.benchmark: {cudnn.benchmark if device.type == 'cuda' else 'N/A'}")
    print("═" * 70)

    phase2_entered = False

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        # ── Phase 2: unfreeze backbone ─────────────────────────────────────
        if not phase2_entered and epoch >= cfg.freeze_epochs:
            print(f"\nEpoch {epoch}: → Phase 2 (full fine-tune)")
            unfreeze_backbone(model)
            optimizer = make_optimizer_phase2(model, cfg)
            remaining = cfg.epochs - epoch
            lr_lambda2 = get_lr_lambda(0, remaining)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda2)
            scaler = GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")
            phase2_entered = True

        train_loss, train_per = run_epoch(
            model, train_loader, optimizer, scaler, device, cfg,
            train=True, accumulation_steps=cfg.accumulation_steps
        )
        val_loss, val_per = run_epoch(
            model, val_loader, optimizer, scaler, device, cfg,
            train=False
        )
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"\nEpoch {epoch + 1:03d}/{cfg.epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        # Per-metric breakdown
        if val_per and target_cols:
            col_w = max(len(c) for c in target_cols)
            for col, v_loss, t_loss in zip(target_cols, val_per, train_per):
                print(f"    {col:<{col_w}}  train={t_loss:.4f}  val={v_loss:.4f}")

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, target_cols, cfg.num_frames, best_ckpt)
            print(f"  ✓ Best model saved → {best_ckpt}")

        # Periodic snapshot
        if (epoch + 1) % 5 == 0:
            snap = ckpt_dir / f"{cfg.experiment_name}_epoch{epoch + 1:03d}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, target_cols, cfg.num_frames, snap)

        # Early stopping
        if early_stop.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1} "
                  f"(no improvement for {cfg.patience} epochs)")
            break

    print(f"\nDone. Best val loss: {best_val_loss:.4f} → {best_ckpt}")


if __name__ == "__main__":
    main()
