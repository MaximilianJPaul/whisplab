from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

from whisplab.config import (
    INACTIVE_CLASS,
    NUM_TECHNIQUES,
    STANDARD_TUNING,
    TECHNIQUE_NAMES,
    Config,
    resolve_device,
)
from whisplab.model import TabTransformer


_TECHNIQUE_SYMBOLS: Dict[str, str] = {
    "hammer": "h",  # hammer-on / pull-off
    "slide": "/",  # slide
    "bend": "b",  # bend
    "vibrato": "~",  # vibrato
    "palm_mute": "PM",  # palm mute
    "dead": "x",  # dead note
    "let_ring": "LR",  # let ring
    "ghost_note": "()",  # ghost note
    "accentuated": ">",  # accent
    "staccato": ".",  # staccato
    "harmonic": "*",  # harmonic
}


def predict_tablature(
    audio_path: Path,
    model: TabTransformer,
    cfg: Config,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != cfg.sample_rate:
        waveform = torchaudio.transforms.Resample(sr, cfg.sample_rate)(waveform)

    waveform = waveform.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(waveform)

    result: Dict[str, torch.Tensor] = {}

    result["fret"] = torch.stack(
        [out.squeeze(0).argmax(dim=-1) for out in outputs["fret"]], dim=-1
    )  # (T, 6)

    if "onset" in outputs:
        result["onset"] = torch.stack(
            [torch.sigmoid(out.squeeze(0).squeeze(-1)) for out in outputs["onset"]],
            dim=-1,
        )  # (T, 6)

    if "technique" in outputs:
        result["technique"] = torch.stack(
            [torch.sigmoid(out.squeeze(0)) for out in outputs["technique"]], dim=1
        )  # (T, 6, 11)

    return result


STRING_NAMES = ["e", "B", "G", "D", "A", "E"]  # high to low


def _format_techniques(tech_probs: Optional[torch.Tensor]) -> str:
    if tech_probs is None:
        return ""

    symbols = []
    for t in range(NUM_TECHNIQUES):
        if tech_probs[t] > 0.5:
            symbols.append(_TECHNIQUE_SYMBOLS[TECHNIQUE_NAMES[t]])
    return "".join(symbols)


def preds_to_tablature_text(
    predictions: Dict[str, torch.Tensor],
    frame_duration: float,
    min_note_frames: int = 3,
) -> str:
    preds = predictions["fret"]
    has_techniques = "technique" in predictions
    tech_preds = predictions.get("technique")  # (T, 6, 11) or None

    T, S = preds.shape
    lines: List[str] = []

    events: List[dict] = []
    for s in range(S):
        col = preds[:, s].tolist()
        i = 0
        while i < T:
            fret = col[i]
            if fret == INACTIVE_CLASS:
                i += 1
                continue
            j = i
            while j < T and col[j] == fret:
                j += 1
            if j - i >= min_note_frames:
                onset = i * frame_duration
                offset = j * frame_duration

                tech_str = ""
                if has_techniques and tech_preds is not None:
                    note_tech = tech_preds[i:j, s, :].mean(dim=0)
                    tech_str = _format_techniques(note_tech)

                events.append(
                    {
                        "string": s,
                        "fret": fret,
                        "onset": onset,
                        "offset": offset,
                        "techniques": tech_str,
                    }
                )
            i = j

    events.sort(key=lambda e: (e["onset"], e["string"]))

    if not events:
        return "(no notes detected)"

    lines.append("# Predicted Guitar Tablature")
    lines.append(f"# Total duration: {T * frame_duration:.1f}s")
    lines.append(f"# {len(events)} note events detected")
    lines.append("")
    lines.append(
        f"{'Time':>8s}  {'String':>6s}  {'Fret':>4s}  {'Duration':>8s}  {'Techniques'}"
    )
    lines.append("-" * 52)
    for ev in events:
        dur = ev["offset"] - ev["onset"]
        tech = ev["techniques"]
        lines.append(
            f"{ev['onset']:8.2f}  {STRING_NAMES[ev['string']]:>6s}  "
            f"{ev['fret']:4d}  {dur:8.2f}s  {tech}"
        )

    return "\n".join(lines)


def preds_to_midi(
    predictions: Dict[str, torch.Tensor],
    frame_duration: float,
    output_path: Path,
    tuning: Tuple[int, ...] = STANDARD_TUNING,
    min_note_frames: int = 3,
) -> None:
    import mido

    preds = predictions["fret"]
    has_techniques = "technique" in predictions
    tech_preds = predictions.get("technique")

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000))  # 120 BPM

    T, S = preds.shape
    tuning_reversed = list(reversed(tuning))  # index 0 = high E

    # Collect note events
    events = []
    for s in range(S):
        col = preds[:, s].tolist()
        i = 0
        while i < T:
            fret = col[i]
            if fret == INACTIVE_CLASS:
                i += 1
                continue
            j = i
            while j < T and col[j] == fret:
                j += 1
            if j - i >= min_note_frames:
                pitch = tuning_reversed[s] + fret
                onset_sec = i * frame_duration
                offset_sec = j * frame_duration

                tech_str = ""
                if has_techniques and tech_preds is not None:
                    note_tech = tech_preds[i:j, s, :].mean(dim=0)
                    tech_str = _format_techniques(note_tech)

                events.append((onset_sec, "on", pitch, tech_str))
                events.append((offset_sec, "off", pitch, ""))
            i = j

    events.sort(key=lambda e: e[0])

    ticks_per_beat = mid.ticks_per_beat
    bpm = 120.0
    sec_per_tick = 60.0 / (bpm * ticks_per_beat)
    prev_tick = 0
    for time_sec, kind, pitch, tech_str in events:
        abs_tick = int(time_sec / sec_per_tick)
        delta = max(0, abs_tick - prev_tick)
        if kind == "on":
            if tech_str:
                track.append(
                    mido.MetaMessage("text", text=f"tech:{tech_str}", time=delta)
                )
                delta = 0
            track.append(mido.Message("note_on", note=pitch, velocity=80, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
        prev_tick = abs_tick

    mid.save(str(output_path))
    print(f"MIDI saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict tablature from audio")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="predictions")
    parser.add_argument("--midi", action="store_true", help="Also output MIDI")
    args = parser.parse_args()

    cfg = Config()
    device = resolve_device()

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else cfg.checkpoint_dir / "best_model.pt"
    )
    model = TabTransformer(cfg).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    audio_path = Path(args.audio)
    predictions = predict_tablature(audio_path, model, cfg, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tab_text = preds_to_tablature_text(predictions, cfg.frame_duration)
    print("\n" + tab_text)

    txt_path = out_dir / f"{audio_path.stem}_tab.txt"
    txt_path.write_text(tab_text)
    print(f"\nTablature saved to {txt_path}")

    if args.midi:
        midi_path = out_dir / f"{audio_path.stem}_tab.mid"
        preds_to_midi(predictions, cfg.frame_duration, midi_path)


if __name__ == "__main__":
    main()
