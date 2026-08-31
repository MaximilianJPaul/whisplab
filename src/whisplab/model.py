"""Transformer-based multi-task AMT model for guitar tablature transcription.

Architecture:
    Waveform → nnAudio CQT → SpecAugment (train only)
    → Multi-Scale CNN Feature Extractor → Linear Projection
    → Sinusoidal Positional Encoding
    → Transformer Encoder (8 layers, 8 heads, Pre-LN)
        ├──→ 6 × Fret Classification MLP Heads       (B, T, 24)
        ├──→ 6 × Onset Detection MLP Heads            (B, T, 1)
        └──→ String Interaction + 6 × Technique MLP   (B, T, 11)

Key design choices:
    • Pre-LayerNorm transformer for stable training with deeper stacks.
    • Multi-scale CNN frontend captures both fine-grained (3×3) and broader
      (5×5) spectro-temporal patterns before the transformer.
    • Sinusoidal positional encoding generalises to longer sequences at
      inference (no learned length limit).
    • Squeeze-and-Excitation (SE) channel attention in CNN blocks.
    • Two-layer MLP heads with GELU activation for richer decision boundaries.
    • A lightweight cross-string transformer layer before technique heads lets
      the model learn interactions across strings (e.g., barre chord bends).
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torchaudio.transforms as T
from nnAudio.features.cqt import CQT1992v2

from whisplab.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x — (B, T, D)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        B, C, _, _ = x.shape
        scale = self.squeeze(x).view(B, C)
        scale = self.excite(scale).view(B, C, 1, 1)
        return x * scale


class _MultiScaleCNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride_freq: int = 1) -> None:
        super().__init__()
        half_ch = out_ch // 2

        self.path_3x3 = nn.Sequential(
            nn.Conv2d(
                in_ch,
                half_ch,
                kernel_size=3,
                stride=(stride_freq, 1),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(half_ch),
            nn.GELU(),
        )

        self.path_5x5 = nn.Sequential(
            nn.Conv2d(
                in_ch,
                half_ch,
                kernel_size=5,
                stride=(stride_freq, 1),
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(half_ch),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(half_ch * 2, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = _SEBlock(out_ch)
        self.act = nn.GELU()

        self.downsample = None
        if stride_freq != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch, kernel_size=1, stride=(stride_freq, 1), bias=False
                ),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        f3 = self.path_3x3(x)
        f5 = self.path_5x5(x)
        out = torch.cat([f3, f5], dim=1)  # (B, out_ch, F', T)
        out = self.fusion(out)

        out = self.conv2(out)
        out = self.se(out)

        out = out + identity
        out = self.act(out)
        return out


class PreLNTransformerLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=True,  # Pre-LayerNorm
            batch_first=True,
        )


class _MLPHead(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabTransformer(nn.Module):
    """Transformer-based multi-task architecture for guitar tablature transcription.

    Input:  (B, 1, T_samples) — raw waveform
    Output: dict with keys:
        "fret":      list of 6 tensors, each (B, T_frames, num_classes)
        "onset":     list of 6 tensors, each (B, T_frames, 1)       [if onset_head]
        "technique": list of 6 tensors, each (B, T_frames, num_tech) [if technique_head]
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

        # Audio Representation (GPU CQT)
        self.cqt = CQT1992v2(
            sr=cfg.sample_rate,
            hop_length=cfg.hop_length,
            fmin=cfg.fmin,
            n_bins=cfg.n_bins,
            bins_per_octave=cfg.bins_per_octave,
            pad_mode="constant",
            trainable=False,
        )

        # SpecAugment
        self.freq_mask = T.FrequencyMasking(freq_mask_param=20)
        self.time_mask = T.TimeMasking(time_mask_param=50)

        # Multi-Scale CNN Feature Extractor
        channels = cfg.cnn_channels  # [1, 32, 64, 128, 256]

        cnn_layers: list[nn.Module] = []

        cnn_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    1, channels[1], kernel_size=7, stride=(2, 1), padding=3, bias=False
                ),
                nn.BatchNorm2d(channels[1]),
                nn.GELU(),
            )
        )

        # Multi-scale CNN blocks (3 blocks)
        for i in range(1, len(channels) - 1):
            cnn_layers.append(
                _MultiScaleCNNBlock(channels[i], channels[i + 1], stride_freq=2)
            )

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate shape after CNN
        # Total freq downsample = 2 (initial) * 2^(len(channels)-2) = 2 * 2^3 = 16
        freq_bins_after_cnn = cfg.n_bins // (2 ** (len(channels) - 1))
        cnn_output_size = channels[-1] * freq_bins_after_cnn

        # Projection + Positional Encoding
        self.projection = nn.Sequential(
            nn.Linear(cnn_output_size, cfg.transformer_dim),
            nn.LayerNorm(cfg.transformer_dim),
        )

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=cfg.transformer_dim,
            max_len=8192,
            dropout=cfg.dropout,
        )

        # Transformer Encoder
        encoder_layer = PreLNTransformerLayer(
            d_model=cfg.transformer_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.transformer_layers,
            norm=nn.LayerNorm(cfg.transformer_dim),  # final norm
            enable_nested_tensor=False,
        )

        # Classification heads (one per string)
        self.output_dropout = nn.Dropout(cfg.dropout)

        # Primary: fret classification (two-layer MLP per string)
        self.fret_heads = nn.ModuleList(
            [
                _MLPHead(
                    cfg.transformer_dim,
                    cfg.head_hidden_dim,
                    cfg.num_classes,
                    cfg.dropout,
                )
                for _ in range(cfg.num_strings)
            ]
        )

        # Auxiliary: onset detection (two-layer MLP per string)
        self.has_onset_head = cfg.onset_head
        if self.has_onset_head:
            self.onset_heads = nn.ModuleList(
                [
                    _MLPHead(cfg.transformer_dim, cfg.head_hidden_dim, 1, cfg.dropout)
                    for _ in range(cfg.num_strings)
                ]
            )

        # Auxiliary: technique classification with cross-string interaction
        self.has_technique_head = cfg.technique_head
        if self.has_technique_head:
            # Cross-string interaction: a small transformer that lets
            # technique predictions on one string attend to other strings.
            # Input: (B*T, 6, D) — treat each frame's 6 strings as a sequence.
            cross_string_layer = nn.TransformerEncoderLayer(
                d_model=cfg.transformer_dim,
                nhead=4,
                dim_feedforward=cfg.transformer_dim * 2,
                dropout=cfg.dropout,
                activation="gelu",
                norm_first=True,
                batch_first=True,
            )
            self.cross_string_transformer = nn.TransformerEncoder(
                cross_string_layer,
                num_layers=2,
                norm=nn.LayerNorm(cfg.transformer_dim),
                enable_nested_tensor=False,
            )

            # Per-string technique heads
            self.technique_heads = nn.ModuleList(
                [
                    _MLPHead(
                        cfg.transformer_dim,
                        cfg.head_hidden_dim,
                        cfg.num_techniques,
                        cfg.dropout,
                    )
                    for _ in range(cfg.num_strings)
                ]
            )

    def forward(self, waveform: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            waveform: (B, 1, T_samples) raw audio

        Returns:
            Dict with keys:
                "fret":      List of 6 tensors, each (B, T_frames, num_classes)
                "onset":     List of 6 tensors, each (B, T_frames, 1)
                "technique": List of 6 tensors, each (B, T_frames, num_techniques)
        """
        # CQT: (B, 1, T_samples) → (B, n_bins, T_frames)
        # nnAudio CQT removes dim=1 if it's 1. So input shape should be (B, T_samples)
        if waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)

        x = self.cqt(waveform)  # (B, n_bins, T_frames)

        # Log scaling
        x = torch.log(x + 1e-5)

        if self.training:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        # CNN expects (B, C, F, T)
        x = x.unsqueeze(1)  # (B, 1, n_bins, T_frames)

        # Multi-scale CNN: (B, 1, n_bins, T) → (B, C, freq', T)
        features = self.cnn(x)

        # Reshape: collapse freq & channels → (B, T, C*freq')
        B, C, F, T_frames = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()  # (B, T, C, F)
        features = features.view(B, T_frames, C * F)

        # Projection + LayerNorm
        features = self.projection(features)  # (B, T, transformer_dim)

        # Positional encoding
        features = self.pos_encoding(features)

        # Transformer Encoder
        transformer_out = self.transformer(features)  # (B, T, transformer_dim)

        transformer_out = self.output_dropout(transformer_out)

        # ── Output heads ───────────────────────────────────────────────
        result: Dict[str, List[torch.Tensor]] = {}

        # Fret classification
        fret_outputs: List[torch.Tensor] = []
        for head in self.fret_heads:
            fret_outputs.append(head(transformer_out))  # (B, T, num_classes)
        result["fret"] = fret_outputs

        # Onset detection
        if self.has_onset_head:
            onset_outputs: List[torch.Tensor] = []
            for head in self.onset_heads:
                onset_outputs.append(head(transformer_out))  # (B, T, 1)
            result["onset"] = onset_outputs

        # Technique classification with cross-string attention
        if self.has_technique_head:
            # Prepare cross-string input: replicate transformer output for
            # each string, stack to (B*T, 6, D), run cross-string attention.
            # Each string gets the same initial representation but the
            # cross-string transformer allows them to exchange information.
            B_out, T_out, D = transformer_out.shape

            # (B, T, D) → (B*T, 1, D) → (B*T, 6, D) via expand
            string_features = transformer_out.reshape(B_out * T_out, 1, D)
            string_features = string_features.expand(
                -1, self.cfg.num_strings, -1
            ).contiguous()

            # Cross-string self-attention
            string_features = self.cross_string_transformer(
                string_features
            )  # (B*T, 6, D)

            # Reshape back: (B*T, 6, D) → (B, T, 6, D)
            string_features = string_features.view(
                B_out, T_out, self.cfg.num_strings, D
            )

            tech_outputs: List[torch.Tensor] = []
            for s, head in enumerate(self.technique_heads):
                # (B, T, D) for string s
                tech_outputs.append(head(string_features[:, :, s, :]))
            result["technique"] = tech_outputs

        return result
