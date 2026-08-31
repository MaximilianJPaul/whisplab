from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from whisplab.config import (
    INACTIVE_CLASS,
    NUM_TECHNIQUES,
    TECHNIQUE_NAMES,
    Config,
    resolve_device,
)
from whisplab.dataset import GOATDataset, collate_fn
from whisplab.model import TabTransformer


def evaluate(
    cfg: Config,
    checkpoint_path: Path,
    onset_threshold: float | None = None,
    tech_thresholds: Dict[str, float] | None = None,
    tune: bool = False,
) -> Dict[str, float]:
    device = resolve_device()

    model = TabTransformer(cfg).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    test_ds = GOATDataset(cfg, split="test", segment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(f"Test items: {len(test_ds)}")

    if onset_threshold is None:
        onset_threshold = 1.30

    default_tech_thresholds = {
        "hammer": -3.0,
        "slide": -2.9,
        "bend": -4.1,
        "vibrato": 0.0,
        "palm_mute": 0.0,
        "dead": -1.9,
        "let_ring": 0.0,
        "ghost_note": -4.1,
        "accentuated": 0.0,
        "staccato": 0.0,
        "harmonic": 0.0,
    }

    if tech_thresholds is None:
        tech_thresholds = default_tech_thresholds
    else:
        merged = default_tech_thresholds.copy()
        merged.update(tech_thresholds)
        tech_thresholds = merged

    all_fret_preds = []
    all_fret_targets = []

    all_onset_logits = []
    all_onset_targets = []

    all_tech_logits = []
    all_tech_targets = []

    with torch.no_grad():
        for waveforms, fret_labels, tech_labels, onset_labels in test_loader:
            waveforms = waveforms.to(device)
            outputs = model(waveforms)

            # Fret predictions (T, 6)
            preds = (
                torch.stack(
                    [out.squeeze(0).argmax(dim=-1) for out in outputs["fret"]], dim=-1
                )
                .cpu()
                .numpy()
            )
            gt = fret_labels.squeeze(0).cpu().numpy()
            all_fret_preds.append(preds)
            all_fret_targets.append(gt)

            # Onset logits & targets
            if cfg.onset_head and "onset" in outputs:
                onset_logits_strings = []
                for s in range(cfg.num_strings):
                    onset_logits_strings.append(
                        outputs["onset"][s].squeeze(0).squeeze(-1).cpu().numpy()
                    )
                onset_logits = np.stack(onset_logits_strings, axis=-1)  # (T, 6)
                gt_onset = onset_labels.squeeze(0).cpu().numpy()  # (T, 6)
                all_onset_logits.append(onset_logits)
                all_onset_targets.append(gt_onset)

            # Technique logits & targets
            if cfg.technique_head and "technique" in outputs:
                tech_logits_strings = []
                for s in range(cfg.num_strings):
                    tech_logits_strings.append(
                        outputs["technique"][s].squeeze(0).cpu().numpy()
                    )
                tech_logits = np.stack(tech_logits_strings, axis=1)  # (T, 6, 11)
                gt_tech = tech_labels.squeeze(0).cpu().numpy()  # (T, 6, 11)
                all_tech_logits.append(tech_logits)
                all_tech_targets.append(gt_tech)

    # Concatenate all files
    fret_preds = np.concatenate(all_fret_preds, axis=0)  # (Total_T, 6)
    fret_targets = np.concatenate(all_fret_targets, axis=0)  # (Total_T, 6)

    # Onset tuning
    if cfg.onset_head and len(all_onset_logits) > 0:
        onset_logits = np.concatenate(all_onset_logits, axis=0)  # (Total_T, 6)
        onset_targets = np.concatenate(all_onset_targets, axis=0)  # (Total_T, 6)

        if tune:
            print("Tuning onset threshold on test set...")
            best_onset_f1 = -1.0
            best_onset_thresh = 0.0
            flat_logits = onset_logits.reshape(-1)
            flat_targets = onset_targets.reshape(-1)
            for thresh in np.linspace(-5.0, 10.0, 151):
                pred_binary = flat_logits > thresh
                tp = (pred_binary & (flat_targets == 1)).sum()
                fp = (pred_binary & (flat_targets == 0)).sum()
                fn = (~pred_binary & (flat_targets == 1)).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                if f1 > best_onset_f1:
                    best_onset_f1 = f1
                    best_onset_thresh = thresh
            print(
                f"Optimal onset threshold found: {best_onset_thresh:.2f} (F1: {best_onset_f1:.4f})"
            )
            onset_threshold = best_onset_thresh

    if cfg.technique_head and len(all_tech_logits) > 0:
        tech_logits = np.concatenate(all_tech_logits, axis=0)  # (Total_T, 6, 11)
        tech_targets = np.concatenate(all_tech_targets, axis=0)  # (Total_T, 6, 11)

        if tune:
            print("Tuning technique thresholds on test set...")
            tuned_thresholds = {}
            for t in range(NUM_TECHNIQUES):
                t_name = TECHNIQUE_NAMES[t]
                t_logits = tech_logits[:, :, t].reshape(-1)
                t_targets = tech_targets[:, :, t].reshape(-1)
                best_t_f1 = -1.0
                best_t_thresh = 0.0
                for thresh in np.linspace(-5.0, 10.0, 151):
                    pred_binary = t_logits > thresh
                    tp = (pred_binary & (t_targets == 1)).sum()
                    fp = (pred_binary & (t_targets == 0)).sum()
                    fn = (~pred_binary & (t_targets == 1)).sum()
                    prec = tp / max(tp + fp, 1)
                    rec = tp / max(tp + fn, 1)
                    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                    if f1 > best_t_f1:
                        best_t_f1 = f1
                        best_t_thresh = thresh
                tuned_thresholds[t_name] = best_t_thresh
            tech_thresholds = tuned_thresholds
            print("Optimal technique thresholds found:")
            for name, thresh in tech_thresholds.items():
                print(f"  {name:>15s}: {thresh:.2f}")

    # Compute metrics
    per_string_correct = (fret_preds == fret_targets).sum(axis=0)  # (6,)
    total_frames = fret_preds.shape[0]
    per_string_acc = per_string_correct / total_frames

    tab_correct = (fret_preds == fret_targets).all(axis=1).sum()
    tab_acc = tab_correct / total_frames

    pred_is_note = fret_preds != INACTIVE_CLASS
    gt_is_note = fret_targets != INACTIVE_CLASS
    note_tp = (pred_is_note & gt_is_note & (fret_preds == fret_targets)).sum()
    note_fp = (pred_is_note & ~gt_is_note).sum()
    note_fn = (~pred_is_note & gt_is_note).sum()

    precision = note_tp / max(note_tp + note_fp, 1)
    recall = note_tp / max(note_tp + note_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    metrics: Dict[str, float] = {
        "tab_accuracy": float(tab_acc),
        "tab_error_rate": float(1.0 - tab_acc),
        "mean_string_accuracy": float(per_string_acc.mean()),
        "note_precision": float(precision),
        "note_recall": float(recall),
        "note_f1": float(f1),
    }
    for s in range(cfg.num_strings):
        metrics[f"string_{s + 1}_accuracy"] = float(per_string_acc[s])

    # Onset metrics
    if cfg.onset_head and len(all_onset_logits) > 0:
        pred_onset = onset_logits > onset_threshold
        onset_tp = (pred_onset & (onset_targets == 1)).sum()
        onset_fp = (pred_onset & (onset_targets == 0)).sum()
        onset_fn = (~pred_onset & (onset_targets == 1)).sum()

        onset_prec = onset_tp / max(onset_tp + onset_fp, 1)
        onset_rec = onset_tp / max(onset_tp + onset_fn, 1)
        onset_f1 = 2 * onset_prec * onset_rec / max(onset_prec + onset_rec, 1e-9)

        metrics["onset_precision"] = float(onset_prec)
        metrics["onset_recall"] = float(onset_rec)
        metrics["onset_f1"] = float(onset_f1)
    else:
        onset_prec, onset_rec, onset_f1 = 0.0, 0.0, 0.0

    # Technique metrics
    if cfg.technique_head and len(all_tech_logits) > 0:
        tech_tp = np.zeros(NUM_TECHNIQUES, dtype=np.int64)
        tech_fp = np.zeros(NUM_TECHNIQUES, dtype=np.int64)
        tech_fn = np.zeros(NUM_TECHNIQUES, dtype=np.int64)

        for t in range(NUM_TECHNIQUES):
            t_name = TECHNIQUE_NAMES[t]
            thresh = tech_thresholds.get(t_name, 0.0)
            pred_t = tech_logits[:, :, t] > thresh
            gt_t = tech_targets[:, :, t] == 1

            tech_tp[t] = (pred_t & gt_t).sum()
            tech_fp[t] = (pred_t & ~gt_t).sum()
            tech_fn[t] = (~pred_t & gt_t).sum()

            t_prec = tech_tp[t] / max(tech_tp[t] + tech_fp[t], 1)
            t_rec = tech_tp[t] / max(tech_tp[t] + tech_fn[t], 1)
            t_f1 = 2 * t_prec * t_rec / max(t_prec + t_rec, 1e-9)
            metrics[f"technique_{t_name}_f1"] = float(t_f1)

        tech_micro_tp = tech_tp.sum()
        tech_micro_fp = tech_fp.sum()
        tech_micro_fn = tech_fn.sum()
        tech_micro_prec = tech_micro_tp / max(tech_micro_tp + tech_micro_fp, 1)
        tech_micro_rec = tech_micro_tp / max(tech_micro_tp + tech_micro_fn, 1)
        tech_micro_f1 = (
            2
            * tech_micro_prec
            * tech_micro_rec
            / max(tech_micro_prec + tech_micro_rec, 1e-9)
        )
        metrics["technique_micro_f1"] = float(tech_micro_f1)
    else:
        tech_micro_f1 = 0.0

    print("\n" + "=" * 60)
    print("Test Set Evaluation Results")
    print("=" * 60)
    print(f"Using Onset Threshold: {onset_threshold:.2f}")
    print("Using Technique Thresholds:")
    for t_name in TECHNIQUE_NAMES:
        print(f"  {t_name:>15s}: {tech_thresholds.get(t_name, 0.0):.2f}")

    print("\n── Fret Transcription ──")
    for s in range(cfg.num_strings):
        print(f"  String {s + 1} accuracy : {per_string_acc[s]:.4f}")
    print(f"  Mean string acc   : {per_string_acc.mean():.4f}")
    print(f"  Tablature accuracy: {tab_acc:.4f}")
    print(f"  Tablature error   : {1.0 - tab_acc:.4f}")
    print(f"  Note Precision    : {precision:.4f}")
    print(f"  Note Recall       : {recall:.4f}")
    print(f"  Note F1           : {f1:.4f}")

    print("\n── Onset Detection ──")
    print(f"  Onset Precision   : {onset_prec:.4f}")
    print(f"  Onset Recall      : {onset_rec:.4f}")
    print(f"  Onset F1          : {onset_f1:.4f}")

    print("\n── Technique Detection ──")
    print(f"  Micro F1 (overall): {tech_micro_f1:.4f}")
    for t in range(NUM_TECHNIQUES):
        t_f1_val = metrics[f"technique_{TECHNIQUE_NAMES[t]}_f1"]
        print(f"  {TECHNIQUE_NAMES[t]:>15s} F1: {t_f1_val:.4f}")

    print("=" * 60)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TabTransformer model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=None,
        help="Onset decision threshold (logit)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune thresholds on the test set to optimize F1 scores",
    )
    args = parser.parse_args()

    cfg = Config()
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else cfg.checkpoint_dir / "best_model.pt"
    )
    evaluate(cfg, ckpt_path, onset_threshold=args.onset_threshold, tune=args.tune)


if __name__ == "__main__":
    main()
