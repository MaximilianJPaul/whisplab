from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from whisplab.config import (
    Config,
    INACTIVE_CLASS,
    NUM_TECHNIQUES,
    TECHNIQUE_NAMES,
    resolve_device,
)
from whisplab.dataset import GOATDataset, collate_fn
from whisplab.model import TabTransformer


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss_raw = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss_raw)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_raw

        if self.weight is not None:
            batch_weights = self.weight[targets]
            focal_loss = focal_loss * batch_weights

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        original = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return original

    def restore(self, model: nn.Module, original: Dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in original:
                param.data.copy_(original[name])


def _compute_class_weights(dataset: GOATDataset, cfg: Config) -> torch.Tensor:
    counts = torch.zeros(cfg.num_classes, dtype=torch.float32)
    # Sample up to 20 items to estimate distribution
    n_sample = min(len(dataset), 20)
    for i in range(n_sample):
        _, fret_labels, _, _ = dataset[i]
        for c in range(cfg.num_classes):
            counts[c] += (fret_labels == c).sum().item()
    total = counts.sum()
    weights = total / (cfg.num_classes * counts.clamp(min=1))
    weights = weights.clamp(min=0.1, max=3.0)
    return weights


def _compute_technique_pos_weights(dataset: GOATDataset, cfg: Config) -> torch.Tensor:
    pos_counts = torch.zeros(cfg.num_techniques, dtype=torch.float32)
    total_counts = torch.zeros(cfg.num_techniques, dtype=torch.float32)
    n_sample = min(len(dataset), 40)  # use a larger sample since techniques are rare
    for i in range(n_sample):
        _, _, tech_labels, _ = dataset[i]  # (T, 6, 11)
        tech_flat = tech_labels.reshape(-1, cfg.num_techniques)
        pos_counts += tech_flat.sum(dim=0)
        total_counts += tech_flat.shape[0]

    neg_counts = total_counts - pos_counts
    weights = neg_counts / pos_counts.clamp(min=1.0)
    # Clamp to reasonable range to avoid exploding gradients
    weights = weights.clamp(min=1.0, max=50.0)
    return weights


def _split_train_val(dataset: GOATDataset, cfg: Config) -> Tuple[Subset, Subset]:
    n = len(dataset)
    n_val = max(1, int(n * cfg.val_fraction))
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _mixup_batch(
    waveforms: torch.Tensor,
    fret_labels: torch.Tensor,
    tech_labels: torch.Tensor,
    onset_labels: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return waveforms, fret_labels, tech_labels, onset_labels, None, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5

    B = waveforms.size(0)
    perm = torch.randperm(B, device=waveforms.device)

    mixed_waveforms = lam * waveforms + (1 - lam) * waveforms[perm]
    mixed_tech = lam * tech_labels + (1 - lam) * tech_labels[perm]
    mixed_onset = lam * onset_labels + (1 - lam) * onset_labels[perm]

    fret_labels_b = fret_labels[perm]

    return mixed_waveforms, fret_labels, fret_labels_b, mixed_tech, mixed_onset, lam


def _compute_technique_f1(
    tech_tp: np.ndarray, tech_fp: np.ndarray, tech_fn: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    micro_tp = tech_tp.sum()
    micro_fp = tech_fp.sum()
    micro_fn = tech_fn.sum()
    micro_prec = micro_tp / max(micro_tp + micro_fp, 1)
    micro_rec = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-9)

    per_tech = {}
    for t in range(NUM_TECHNIQUES):
        t_prec = tech_tp[t] / max(tech_tp[t] + tech_fp[t], 1)
        t_rec = tech_tp[t] / max(tech_tp[t] + tech_fn[t], 1)
        t_f1 = 2 * t_prec * t_rec / max(t_prec + t_rec, 1e-9)
        per_tech[TECHNIQUE_NAMES[t]] = float(t_f1)

    return float(micro_f1), per_tech


def train(cfg: Config) -> None:
    device = resolve_device()
    print(f"Device: {device}")

    full_train_ds = GOATDataset(cfg, split="train", segment=True)
    train_ds, val_ds = _split_train_val(full_train_ds, cfg)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = TabTransformer(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} ({trainable_params:,} trainable)")

    class_weights = _compute_class_weights(full_train_ds, cfg).to(device)
    fret_criterion = FocalLoss(
        weight=class_weights,
        gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
    )
    onset_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(cfg.onset_pos_weight, device=device)
    )

    technique_pos_weight = _compute_technique_pos_weights(full_train_ds, cfg).to(device)
    print("Computed per-technique positive weights:")
    for t in range(cfg.num_techniques):
        print(f"  {TECHNIQUE_NAMES[t]:>15s}: {technique_pos_weight[t].item():.2f}")
    technique_criterion = nn.BCEWithLogitsLoss(pos_weight=technique_pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    total_steps = cfg.max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        total_steps=total_steps,
        pct_start=cfg.warmup_epochs / cfg.max_epochs,
        anneal_strategy="cos",
        div_factor=25.0,  # initial_lr = max_lr / 25
        final_div_factor=1000.0,  # final_lr = initial_lr / 1000
    )

    ema = EMA(model, decay=0.999)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    accum_steps = cfg.grad_accumulation_steps

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        train_fret_correct = 0
        train_fret_total = 0
        optimizer.zero_grad()

        for batch_idx, (waveforms, fret_labels, tech_labels, onset_labels) in enumerate(
            train_loader
        ):
            waveforms = waveforms.to(device)
            fret_labels = fret_labels.to(device)
            tech_labels = tech_labels.to(device)
            onset_labels = onset_labels.to(device)

            fret_labels_b = None
            lam = 1.0
            if cfg.mixup_alpha > 0 and epoch > cfg.warmup_epochs:
                (
                    waveforms,
                    fret_labels,
                    fret_labels_b,
                    tech_labels,
                    onset_labels,
                    lam,
                ) = _mixup_batch(
                    waveforms, fret_labels, tech_labels, onset_labels, cfg.mixup_alpha
                )

            outputs = model(waveforms)

            fret_loss = torch.tensor(0.0, device=device)
            for s in range(cfg.num_strings):
                pred = outputs["fret"][s].reshape(-1, cfg.num_classes)
                target_a = fret_labels[:, :, s].reshape(-1)

                if fret_labels_b is not None:
                    target_b = fret_labels_b[:, :, s].reshape(-1)
                    fret_loss = fret_loss + (
                        lam * fret_criterion(pred, target_a)
                        + (1 - lam) * fret_criterion(pred, target_b)
                    )
                else:
                    fret_loss = fret_loss + fret_criterion(pred, target_a)

                train_fret_correct += (pred.argmax(dim=-1) == target_a).sum().item()
                train_fret_total += target_a.numel()

            onset_loss = torch.tensor(0.0, device=device)
            if cfg.onset_head and "onset" in outputs:
                for s in range(cfg.num_strings):
                    pred_onset = outputs["onset"][s].reshape(-1)
                    target_onset = onset_labels[:, :, s].reshape(-1)
                    onset_loss = onset_loss + onset_criterion(pred_onset, target_onset)

            technique_loss = torch.tensor(0.0, device=device)
            if cfg.technique_head and "technique" in outputs:
                for s in range(cfg.num_strings):
                    pred_tech = outputs["technique"][s].reshape(-1, cfg.num_techniques)
                    target_tech = tech_labels[:, :, s, :].reshape(
                        -1, cfg.num_techniques
                    )
                    technique_loss = technique_loss + technique_criterion(
                        pred_tech, target_tech
                    )

            loss = (
                fret_loss
                + cfg.onset_loss_weight * onset_loss
                + cfg.technique_loss_weight * technique_loss
            )

            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(
                train_loader
            ):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        train_acc = train_fret_correct / max(train_fret_total, 1)

        original_weights = ema.apply(model)
        model.eval()
        val_loss = 0.0
        val_fret_correct = 0
        val_fret_total = 0
        note_tp = 0
        note_fp = 0
        note_fn = 0

        active_correct = 0
        active_total = 0

        tech_tp = np.zeros(NUM_TECHNIQUES, dtype=np.int64)
        tech_fp = np.zeros(NUM_TECHNIQUES, dtype=np.int64)
        tech_fn = np.zeros(NUM_TECHNIQUES, dtype=np.int64)

        with torch.no_grad():
            for waveforms, fret_labels, tech_labels, onset_labels in val_loader:
                waveforms = waveforms.to(device)
                fret_labels = fret_labels.to(device)
                tech_labels = tech_labels.to(device)
                onset_labels = onset_labels.to(device)

                outputs = model(waveforms)

                batch_loss = torch.tensor(0.0, device=device)
                for s in range(cfg.num_strings):
                    pred_s = outputs["fret"][s].reshape(-1, cfg.num_classes)
                    target_s = fret_labels[:, :, s].reshape(-1)
                    batch_loss = batch_loss + fret_criterion(pred_s, target_s)

                    pred_cls = pred_s.argmax(dim=-1)
                    # Raw frame accuracy (includes inactive frames — inflated)
                    val_fret_correct += (pred_cls == target_s).sum().item()
                    val_fret_total += target_s.numel()

                    # Active-note accuracy (only frames with a real note)
                    active_mask = target_s != INACTIVE_CLASS
                    if active_mask.any():
                        active_correct += (
                            (pred_cls[active_mask] == target_s[active_mask])
                            .sum()
                            .item()
                        )
                        active_total += active_mask.sum().item()

                    # Note-level P/R/F1 components
                    pred_is_note = pred_cls != INACTIVE_CLASS
                    gt_is_note = active_mask
                    note_tp += (
                        (pred_is_note & gt_is_note & (pred_cls == target_s))
                        .sum()
                        .item()
                    )
                    note_fp += (pred_is_note & ~gt_is_note).sum().item()
                    note_fn += (~pred_is_note & gt_is_note).sum().item()

                # Onset validation loss
                if cfg.onset_head and "onset" in outputs:
                    for s in range(cfg.num_strings):
                        pred_onset = outputs["onset"][s].reshape(-1)
                        target_onset = onset_labels[:, :, s].reshape(-1)
                        batch_loss = (
                            batch_loss
                            + cfg.onset_loss_weight
                            * onset_criterion(pred_onset, target_onset)
                        )

                # Technique validation loss + F1
                if cfg.technique_head and "technique" in outputs:
                    for s in range(cfg.num_strings):
                        pred_tech = outputs["technique"][s].reshape(
                            -1, cfg.num_techniques
                        )
                        target_tech = tech_labels[:, :, s, :].reshape(
                            -1, cfg.num_techniques
                        )
                        batch_loss = (
                            batch_loss
                            + cfg.technique_loss_weight
                            * technique_criterion(pred_tech, target_tech)
                        )

                        # Track technique F1
                        pred_binary = (pred_tech > 0).long()
                        target_binary = target_tech.long()
                        for t in range(NUM_TECHNIQUES):
                            tech_tp[t] += (
                                ((pred_binary[:, t] == 1) & (target_binary[:, t] == 1))
                                .sum()
                                .item()
                            )
                            tech_fp[t] += (
                                ((pred_binary[:, t] == 1) & (target_binary[:, t] == 0))
                                .sum()
                                .item()
                            )
                            tech_fn[t] += (
                                ((pred_binary[:, t] == 0) & (target_binary[:, t] == 1))
                                .sum()
                                .item()
                            )

                val_loss += batch_loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        # Raw frame accuracy (kept for reference; inflated by silent frames)
        val_acc = val_fret_correct / max(val_fret_total, 1)
        # Active-note accuracy: correctness only where a real note is played
        active_acc = active_correct / max(active_total, 1)
        # Note-level F1
        note_prec = note_tp / max(note_tp + note_fp, 1)
        note_rec = note_tp / max(note_tp + note_fn, 1)
        note_f1 = 2 * note_prec * note_rec / max(note_prec + note_rec, 1e-9)

        # Technique F1
        tech_micro_f1, per_tech_f1 = _compute_technique_f1(tech_tp, tech_fp, tech_fn)

        ema.restore(model, original_weights)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{cfg.max_epochs} — "
            f"loss={avg_train_loss:.4f}→{avg_val_loss:.4f}  "
            f"frame_acc={val_acc:.3f}  active_acc={active_acc:.3f}  "
            f"note_f1={note_f1:.3f}  tech_f1={tech_micro_f1:.3f}  "
            f"lr={lr_now:.2e}  {elapsed:.0f}s"
        )

        # Log per-technique F1 every 10 epochs
        if epoch % 10 == 0:
            print("  Technique F1 breakdown:")
            for tech_name, f1_val in per_tech_f1.items():
                print(f"    {tech_name:>15s}: {f1_val:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = cfg.checkpoint_dir / "best_model.pt"
            # Save EMA weights as the checkpoint
            ema_weights = ema.apply(model)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_frame_acc": val_acc,
                    "val_active_acc": active_acc,
                    "val_note_f1": note_f1,
                    "tech_micro_f1": tech_micro_f1,
                    "config": cfg,
                },
                ckpt_path,
            )
            ema.restore(model, ema_weights)
            print(
                f"  ✓ Best model saved "
                f"(active_acc={active_acc:.3f}  note_f1={note_f1:.3f}  "
                f"tech_f1={tech_micro_f1:.3f}  val_loss={avg_val_loss:.4f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(
                    f"  Early stopping after {epoch} epochs (no improvement for {cfg.patience} epochs)"
                )
                break

    print(
        f"\nTraining complete. Best val_loss={best_val_loss:.4f}, best val_acc={best_val_acc:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TabTransformer model")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    train(cfg)


if __name__ == "__main__":
    main()
