import random

import pytest
import torch
import torchaudio

from whisplab.config import INACTIVE_CLASS, NUM_FRETS, NUM_STRINGS, Config
from whisplab.dataset import WaveformAugmentation

SAMPLE_RATE = 22050


def _dominant_frequency(waveform: torch.Tensor, sample_rate: int) -> float:
    spectrum = torch.fft.rfft(waveform.squeeze(0))
    freqs = torch.fft.rfftfreq(waveform.shape[-1], d=1.0 / sample_rate)
    return freqs[spectrum.abs().argmax()].item()


def test_pitch_shift_positive_steps_raises_frequency():
    t = torch.arange(SAMPLE_RATE) / SAMPLE_RATE
    sine_440 = torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)

    shifted = torchaudio.functional.pitch_shift(sine_440, SAMPLE_RATE, 2)

    f0 = _dominant_frequency(shifted, SAMPLE_RATE)
    expected = 440.0 * 2 ** (2 / 12)  # ~493.9 Hz
    assert f0 == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("shift", [2, -2])
def test_fret_labels_move_with_pitch_shift(monkeypatch, shift):
    cfg = Config()
    aug = WaveformAugmentation(cfg)

    # Force the augmentation to always apply this exact shift.
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randint", lambda a, b: shift)

    t = torch.arange(SAMPLE_RATE) / SAMPLE_RATE
    waveform = torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)

    n_frames = 10
    fret_labels = torch.full((n_frames, NUM_STRINGS), INACTIVE_CLASS, dtype=torch.long)
    fret_labels[:, 1] = 5  # fret 5 on the B string

    shifted_wave, shifted_labels, _ = aug(waveform, fret_labels.clone())

    # Audio moved by `shift` semitones, so the label must move by +shift.
    assert (shifted_labels[:, 1] == 5 + shift).all()
    # Inactive strings stay inactive.
    assert (shifted_labels[:, 0] == INACTIVE_CLASS).all()

    # The audio direction matches: positive shift -> higher frequency.
    f0 = _dominant_frequency(shifted_wave, SAMPLE_RATE)
    expected = 440.0 * 2 ** (shift / 12)
    assert f0 == pytest.approx(expected, rel=0.05)


def test_out_of_range_frets_marked_inactive(monkeypatch):
    cfg = Config()
    aug = WaveformAugmentation(cfg)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randint", lambda a, b: 2)

    waveform = torch.sin(
        2 * torch.pi * 440.0 * torch.arange(SAMPLE_RATE) / SAMPLE_RATE
    ).unsqueeze(0)

    fret_labels = torch.full((4, NUM_STRINGS), INACTIVE_CLASS, dtype=torch.long)
    fret_labels[:, 0] = NUM_FRETS - 1  # top fret: +2 pushes it out of range

    _, shifted_labels, _ = aug(waveform, fret_labels)

    assert (shifted_labels[:, 0] == INACTIVE_CLASS).all()
