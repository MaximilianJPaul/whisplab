from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from whisplab.config import (
    INACTIVE_CLASS,
    NUM_FRETS,
    NUM_STRINGS,
    NUM_TECHNIQUES,
    TUNING_PRESETS,
    Config,
)
from whisplab.technique_labels import (
    align_techniques_to_midi,
    onset_labels_from_frets,
    parse_dadagp_techniques,
    techniques_to_frame_labels,
)


def midi_pitch_to_string_fret(
    pitch: int,
    open_strings: Tuple[int, ...],
    max_fret: int = NUM_FRETS - 1,
) -> Optional[Tuple[int, int]]:
    for str_idx, open_pitch in enumerate(reversed(open_strings)):
        fret = pitch - open_pitch
        if 0 <= fret <= max_fret:
            return str_idx, fret
    return None


def parse_midi_notes(
    midi_path: Path,
    tuning: Tuple[int, ...],
) -> List[Dict]:
    mid = mido.MidiFile(str(midi_path))
    notes: List[Dict] = []

    for track in mid.tracks:
        current_time = 0.0  # seconds
        active: Dict[int, float] = {}  # pitch → onset time

        for msg in track:
            # Accumulate delta time → seconds
            current_time += mido.tick2second(
                msg.time, mid.ticks_per_beat, _get_tempo(mid)
            )

            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = current_time
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                onset = active.pop(msg.note, None)
                if onset is not None:
                    sf_result = midi_pitch_to_string_fret(msg.note, tuning)
                    if sf_result is not None:
                        string_idx, fret = sf_result
                        notes.append(
                            {
                                "onset": onset,
                                "offset": current_time,
                                "string": string_idx,
                                "fret": fret,
                            }
                        )
    return notes


def _get_tempo(mid: mido.MidiFile) -> int:
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return msg.tempo
    return 500_000  # default 120 BPM


def notes_to_frame_labels(
    notes: List[Dict],
    num_frames: int,
    frame_duration: float,
) -> torch.Tensor:
    labels = torch.full((num_frames, NUM_STRINGS), INACTIVE_CLASS, dtype=torch.long)

    for note in notes:
        start_frame = int(note["onset"] / frame_duration)
        end_frame = int(note["offset"] / frame_duration)
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        string_idx = note["string"]
        fret = note["fret"]
        if 0 <= string_idx < NUM_STRINGS and 0 <= fret < NUM_FRETS:
            labels[start_frame:end_frame, string_idx] = fret

    return labels


class WaveformAugmentation:
    def __init__(self, cfg: Config) -> None:
        self.pitch_shift_range = cfg.pitch_shift_range
        self.sample_rate = cfg.sample_rate

    def __call__(
        self,
        waveform: torch.Tensor,
        fret_labels: Optional[torch.Tensor] = None,
        technique_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Random gain (±6 dB)
        gain_db = (torch.rand(1) * 12 - 6).item()
        gain_linear = 10 ** (gain_db / 20.0)
        waveform = waveform * gain_linear

        if self.pitch_shift_range > 0 and random.random() < 0.3:
            shift = random.randint(-self.pitch_shift_range, self.pitch_shift_range)
            if shift != 0:
                waveform = torchaudio.functional.pitch_shift(
                    waveform, self.sample_rate, shift
                )
                if fret_labels is not None:
                    active_mask = fret_labels != INACTIVE_CLASS
                    shifted_frets = fret_labels.clone()
                    shifted_frets[active_mask] = shifted_frets[active_mask] + shift
                    # Clamp: if shifted frets go out of range, mark inactive
                    out_of_range = (shifted_frets < 0) | (shifted_frets >= NUM_FRETS)
                    shifted_frets[out_of_range & active_mask] = INACTIVE_CLASS
                    # Also zero out technique labels for invalidated notes
                    if technique_labels is not None:
                        invalid_frames = out_of_range & active_mask
                        technique_labels[invalid_frames] = 0.0
                    fret_labels = shifted_frets

        return waveform, fret_labels, technique_labels


class GOATDataset(Dataset):
    AMP_COLUMNS = [
        "amp_audio_path_1",
        "amp_audio_path_2",
        "amp_audio_path_3",
        "amp_audio_path_4",
        "amp_audio_path_5",
    ]

    def __init__(
        self,
        cfg: Config,
        split: str = "train",
        segment: bool = True,
        use_amp_variants: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.is_training = segment and (split == "train")
        self.segment = self.is_training
        self.augment = WaveformAugmentation(cfg) if self.is_training else None

        df = pd.read_csv(cfg.metadata_csv)
        df = df[df["split"] == split]
        df = df[df["tuning_class"] != "othertunings"]
        df = df[
            df["finealigned_midi_path"].notna() & (df["finealigned_midi_path"] != "")
        ]
        df = df.reset_index(drop=True)

        self._samples: list[tuple[int, str]] = []
        for i in range(len(df)):
            self._samples.append((i, "di_audio_path"))
            if use_amp_variants and self.is_training:
                for col in self.AMP_COLUMNS:
                    self._samples.append((i, col))

        self.items = df

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row_idx, audio_col = self._samples[idx]
        row = self.items.iloc[row_idx]

        raw_audio_path = row[audio_col]
        audio_path = self.cfg.project_root / raw_audio_path.replace(
            "GOAT/", "GOAT/data/", 1
        )
        audio_data, sr = sf.read(str(audio_path), dtype="float32")
        if audio_data.ndim == 1:
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_data.T).mean(dim=0, keepdim=True)

        if sr != self.cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.cfg.sample_rate)
            waveform = resampler(waveform)

        num_samples = waveform.shape[-1]

        num_frames = num_samples // self.cfg.hop_length + 1

        raw_midi_path = row["finealigned_midi_path"]
        midi_path = self.cfg.project_root / raw_midi_path.replace(
            "GOAT/", "GOAT/data/", 1
        )
        tuning_key = row["tuning_class"]
        tuning = TUNING_PRESETS.get(tuning_key, TUNING_PRESETS["standard"])
        midi_notes = parse_midi_notes(midi_path, tuning)
        fret_labels = notes_to_frame_labels(
            midi_notes, num_frames, self.cfg.frame_duration
        )

        # Technique Labels from DaDaGP
        raw_dadagp_path = row.get("dadagp_path", "")
        if self.cfg.technique_head and pd.notna(raw_dadagp_path) and raw_dadagp_path:
            dadagp_path = self.cfg.project_root / raw_dadagp_path.replace(
                "GOAT/", "GOAT/data/", 1
            )
            if dadagp_path.exists():
                dadagp_notes, _tempo, _tpb = parse_dadagp_techniques(dadagp_path)
                notes_with_tech = align_techniques_to_midi(dadagp_notes, midi_notes)
                technique_labels = techniques_to_frame_labels(
                    notes_with_tech, num_frames, self.cfg.frame_duration
                )
            else:
                technique_labels = torch.zeros(
                    num_frames, NUM_STRINGS, NUM_TECHNIQUES, dtype=torch.float32
                )
        else:
            technique_labels = torch.zeros(
                num_frames, NUM_STRINGS, NUM_TECHNIQUES, dtype=torch.float32
            )

        if self.segment:
            seg_frames = self.cfg.segment_frames
            seg_samples = (seg_frames - 1) * self.cfg.hop_length

            if num_samples > seg_samples:
                start_sample = random.randint(0, num_samples - seg_samples)
                start_frame = start_sample // self.cfg.hop_length
                start_sample = start_frame * self.cfg.hop_length

                waveform = waveform[:, start_sample : start_sample + seg_samples]
                fret_labels = fret_labels[start_frame : start_frame + seg_frames]
                technique_labels = technique_labels[
                    start_frame : start_frame + seg_frames
                ]
            else:
                pad_len_samples = seg_samples - num_samples
                waveform = torch.nn.functional.pad(waveform, (0, pad_len_samples))
                pad_len_frames = seg_frames - num_frames
                fret_labels = torch.nn.functional.pad(
                    fret_labels, (0, 0, 0, pad_len_frames), value=INACTIVE_CLASS
                )
                technique_labels = torch.nn.functional.pad(
                    technique_labels, (0, 0, 0, 0, 0, pad_len_frames), value=0.0
                )

        if self.augment is not None:
            waveform, fret_labels, technique_labels = self.augment(
                waveform, fret_labels, technique_labels
            )

        onset_labels = onset_labels_from_frets(
            fret_labels, smear_frames=self.cfg.onset_smear_frames
        )

        return waveform, fret_labels, technique_labels, onset_labels


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    waveforms, fret_list, tech_list, onset_list = zip(*batch)

    max_samples = max(w.shape[-1] for w in waveforms)
    max_frames = max(lbl.shape[0] for lbl in fret_list)

    padded_waveforms = []
    padded_frets = []
    padded_techs = []
    padded_onsets = []

    for w, fret, tech, onset in zip(waveforms, fret_list, tech_list, onset_list):
        pad_s = max_samples - w.shape[-1]
        padded_waveforms.append(torch.nn.functional.pad(w, (0, pad_s)))

        pad_f = max_frames - fret.shape[0]
        padded_frets.append(
            torch.nn.functional.pad(fret, (0, 0, 0, pad_f), value=INACTIVE_CLASS)
        )
        padded_techs.append(
            torch.nn.functional.pad(tech, (0, 0, 0, 0, 0, pad_f), value=0.0)
        )
        padded_onsets.append(
            torch.nn.functional.pad(onset, (0, 0, 0, pad_f), value=0.0)
        )

    return (
        torch.stack(padded_waveforms),
        torch.stack(padded_frets),
        torch.stack(padded_techs),
        torch.stack(padded_onsets),
    )
