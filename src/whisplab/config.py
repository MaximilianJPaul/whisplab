from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def resolve_device() -> torch.device:
    """Pick the best available accelerator: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Standard guitar tuning MIDI pitches: E2=40, A2=45, D3=50, G3=55, B3=59, E4=64
STANDARD_TUNING: Tuple[int, ...] = (40, 45, 50, 55, 59, 64)

# Tuning presets — open-string MIDI pitches for strings 6→1 (low→high)
TUNING_PRESETS: Dict[str, Tuple[int, ...]] = {
    # Standard: E A D G B E
    "standard": (40, 45, 50, 55, 59, 64),
    # Drop D: D A D G B E
    "dropd": (38, 45, 50, 55, 59, 64),
    # Eb Ab Db Gb Bb Eb  (each string 1 semitone down)
    "downtuned": (39, 44, 49, 54, 58, 63),
}

NUM_STRINGS: int = 6
NUM_FRETS: int = 23  # frets 0–22
NUM_CLASSES: int = NUM_FRETS + 1  # 24 = frets 0-22 + "inactive" class (23)
INACTIVE_CLASS: int = 23

TECHNIQUE_NAMES: List[str] = [
    "hammer",  # 0  hammer-on / pull-off / legato
    "slide",  # 1  any slide type
    "bend",  # 2  any bend type
    "vibrato",  # 3  vibrato
    "palm_mute",  # 4  palm muting
    "dead",  # 5  dead / muted note
    "let_ring",  # 6  let ring
    "ghost_note",  # 7  ghost note
    "accentuated",  # 8  accentuated note
    "staccato",  # 9  staccato
    "harmonic",  # 10 natural / artificial harmonics
]
NUM_TECHNIQUES: int = len(TECHNIQUE_NAMES)


@dataclass
class Config:
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    goat_root: Path = field(default=None)  # type: ignore[assignment]
    metadata_csv: Path = field(default=None)  # type: ignore[assignment]
    checkpoint_dir: Path = field(default=None)  # type: ignore[assignment]

    sample_rate: int = 22050
    original_sample_rate: int = 44100
    hop_length: int = 512
    fmin: float = 32.70  # C1, well below the guitar's lowest E2 (82.41 Hz)
    n_bins: int = 192  # Total CQT bins
    bins_per_octave: int = 24  # 2 bins per semitone for sharp harmonic resolution

    segment_duration: float = (
        10.0  # seconds per training crop (increased for more context)
    )

    cnn_channels: List[int] = field(default_factory=lambda: [1, 32, 64, 128, 256])

    transformer_dim: int = 256  # d_model for the transformer
    transformer_heads: int = 8  # number of attention heads
    transformer_layers: int = 8  # number of encoder layers
    transformer_ffn_dim: int = 1024  # feed-forward network inner dimension
    transformer_pre_ln: bool = True  # use Pre-LayerNorm (more stable)
    dropout: float = 0.2

    head_hidden_dim: int = 128  # hidden dim for 2-layer MLP heads
    num_strings: int = NUM_STRINGS
    num_classes: int = NUM_CLASSES
    num_techniques: int = NUM_TECHNIQUES

    onset_head: bool = True  # auxiliary onset detection head
    technique_head: bool = True  # auxiliary technique classification head

    batch_size: int = 8
    learning_rate: float = 3e-4  # slightly higher for transformers w/ OneCycleLR
    focal_gamma: float = 2.0  # Focal loss parameter for heavily imbalanced classes
    label_smoothing: float = 0.05  # label smoothing for fret loss calibration
    max_epochs: int = 120
    patience: int = 20  # early-stopping patience
    val_fraction: float = 0.15  # fraction of train set used for validation
    num_workers: int = 4
    weight_decay: float = 1e-2  # higher weight decay for transformer regularization

    onset_loss_weight: float = 0.5  # weight for onset detection loss
    technique_loss_weight: float = (
        0.4  # weight for technique classification loss (increased)
    )

    technique_pos_weight: float = 5.0  # upweight positive samples for rare techniques

    onset_pos_weight: float = 10.0  # typical ratio of silent:onset frames is ~50:1

    onset_smear_frames: int = 2  # 0 = no smearing; 2 = ±2 frames (~46 ms at 22050/512)

    mixup_alpha: float = 0.3  # Beta distribution α for mixup (0 = disabled)
    pitch_shift_range: int = 2  # ±semitones for pitch-shift augmentation

    warmup_epochs: int = 5  # linear LR warmup epochs
    grad_accumulation_steps: int = 4  # effective batch size = batch_size × this

    def __post_init__(self) -> None:
        if self.goat_root is None:
            self.goat_root = self.project_root / "GOAT"
        if self.metadata_csv is None:
            self.metadata_csv = self.goat_root / "metadata.csv"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "checkpoints"

    @property
    def segment_frames(self) -> int:
        return int(self.segment_duration * self.sample_rate / self.hop_length)

    @property
    def frame_duration(self) -> float:
        return self.hop_length / self.sample_rate
