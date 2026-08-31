from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

from whisplab.config import (
    INACTIVE_CLASS,
    NUM_STRINGS,
    NUM_TECHNIQUES,
    TECHNIQUE_NAMES,
)


_NFX_TAG_MAP: Dict[str, str] = {
    "hammer": "hammer",
    "slide": "slide",
    "bend": "bend",
    "vibrato": "vibrato",
    "palm_mute": "palm_mute",
    "dead": "dead",
    "let_ring": "let_ring",
    "ghost_note": "ghost_note",
    "accentuated_note": "accentuated",
    "staccato": "staccato",
    "harmonic": "harmonic",
}

_TECHNIQUE_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(TECHNIQUE_NAMES)}


def _parse_nfx_tag(tag: str) -> Optional[str]:
    parts = tag.split(":")
    if len(parts) < 2:
        return None
    raw_name = parts[1]
    return _NFX_TAG_MAP.get(raw_name)


def parse_dadagp_techniques(
    txt_path: Path,
) -> Tuple[List[Dict], int, int]:
    ticks_per_beat = 960

    lines = Path(txt_path).read_text().strip().splitlines()

    tempo = 120  # default
    notes: List[Dict] = []
    current_tick = 0

    pending_notes: List[Dict] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("tempo:"):
            tempo = int(line.split(":")[1])
            continue
        if line in ("goat_dataset", "start", "end") or line.startswith("downtune:"):
            continue
        if line == "new_measure":
            continue

        if line.startswith("wait:"):
            current_tick += int(line.split(":")[1])
            continue

        m = re.match(r"^[a-zA-Z0-9_]+:note:s(\d+):f(\d+)$", line)
        if m:
            string_1based = int(m.group(1))
            fret = int(m.group(2))
            note = {
                "onset_tick": current_tick,
                "string": string_1based - 1,  # convert to 0-indexed
                "fret": fret,
                "techniques": set(),
            }
            pending_notes.append(note)
            notes.append(note)
            continue

        if line.endswith(":rest"):
            continue

        if line.startswith("nfx:"):
            tech_name = _parse_nfx_tag(line)
            if tech_name is not None and pending_notes:
                pending_notes[-1]["techniques"].add(tech_name)
            continue

    return notes, tempo, ticks_per_beat


def align_techniques_to_midi(
    dadagp_notes: List[Dict],
    midi_notes: List[Dict],
) -> List[Dict]:
    from collections import defaultdict

    result = []
    for midi_note in midi_notes:
        result.append({**midi_note, "techniques": set()})

    dadagp_by_key: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    for note in dadagp_notes:
        key = (note["string"], note["fret"])
        dadagp_by_key[key].append(note)

    midi_by_key: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    midi_sorted_indices = sorted(
        range(len(midi_notes)), key=lambda i: midi_notes[i]["onset"]
    )
    for idx in midi_sorted_indices:
        n = midi_notes[idx]
        key = (n["string"], n["fret"])
        midi_by_key[key].append(idx)

    for key, dg_list in dadagp_by_key.items():
        midi_indices = midi_by_key.get(key, [])
        for i, dg_note in enumerate(dg_list):
            if i < len(midi_indices) and dg_note["techniques"]:
                result_idx = midi_indices[i]
                result[result_idx]["techniques"] = dg_note["techniques"]

    return result


def techniques_to_frame_labels(
    notes_with_techniques: List[Dict],
    num_frames: int,
    frame_duration: float,
) -> torch.Tensor:
    labels = torch.zeros(num_frames, NUM_STRINGS, NUM_TECHNIQUES, dtype=torch.float32)

    for note in notes_with_techniques:
        if not note["techniques"]:
            continue

        start_frame = max(0, int(note["onset"] / frame_duration))
        end_frame = min(num_frames, int(note["offset"] / frame_duration))
        string_idx = note["string"]

        if not (0 <= string_idx < NUM_STRINGS):
            continue

        for tech_name in note["techniques"]:
            tech_idx = _TECHNIQUE_TO_IDX.get(tech_name)
            if tech_idx is not None:
                labels[start_frame:end_frame, string_idx, tech_idx] = 1.0

    return labels


def onset_labels_from_frets(
    fret_labels: torch.Tensor,
    smear_frames: int = 0,
) -> torch.Tensor:
    T, S = fret_labels.shape
    onsets = torch.zeros(T, S, dtype=torch.float32)

    for s in range(S):
        col = fret_labels[:, s]
        is_active = col != INACTIVE_CLASS

        # Frame 0: onset if active
        if T > 0 and is_active[0]:
            onsets[0, s] = 1.0

        # Frames 1..T-1: onset if active AND different from previous
        if T > 1:
            changed = col[1:] != col[:-1]
            onsets[1:, s] = (is_active[1:] & changed).float()

    if smear_frames <= 0:
        return onsets

    # Triangular smearing: convolve with a triangular kernel
    # kernel = [1/W, 2/W, ..., 1, ..., 2/W, 1/W]  (peak at centre = 1.0)
    W = smear_frames
    kernel_vals = [max(0.0, 1.0 - abs(k) / W) for k in range(-W, W + 1)]
    kernel = torch.tensor(kernel_vals, dtype=torch.float32, device=fret_labels.device)
    kernel = kernel.view(1, 1, -1)  # (1, 1, kernel_size) for conv1d

    # onsets: (T, S) → (S, 1, T) for conv1d → (S, 1, T) → (T, S)
    x = onsets.T.unsqueeze(1)  # (S, 1, T)
    smeared = torch.nn.functional.conv1d(x, kernel, padding=W)  # (S, 1, T)
    smeared = smeared.squeeze(1).T  # (T, S)
    return smeared.clamp(0.0, 1.0)
