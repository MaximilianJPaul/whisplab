from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import guitarpro

from whisplab.config import STANDARD_TUNING
from whisplab.predict import STRING_NAMES

# String 1 is the high E; index 0 of this list corresponds to string 1.
_STRING_PITCHES: List[int] = list(reversed(list(STANDARD_TUNING)))

DEFAULT_BPM = 100.0
DEFAULT_ONSET_TOLERANCE = 0.100  # seconds


def load_hand_transcription(path: Path, bpm: float = DEFAULT_BPM) -> List[Dict]:
    song = guitarpro.parse(str(path))
    track = song.tracks[0]
    tuning = [s.value for s in track.strings]  # index 0 = string 1 (high E)
    seconds_per_quarter = 60.0 / bpm

    notes: List[Dict] = []
    measure_start = 0.0
    for measure in track.measures:
        cursor = measure_start
        for beat in measure.voices[0].beats:
            duration = beat.duration
            quarters = 4.0 / duration.value
            if duration.isDotted:
                quarters *= 1.5
            if duration.tuplet is not None:
                quarters *= duration.tuplet.times / duration.tuplet.enters
            for note in beat.notes:
                if note.type is guitarpro.NoteType.tie:
                    continue
                notes.append(
                    {
                        "time": cursor,
                        "string": note.string,
                        "fret": note.value,
                        "pitch": tuning[note.string - 1] + note.value,
                    }
                )
            cursor += quarters * seconds_per_quarter
        measure_start = cursor

    notes.sort(key=lambda n: (n["time"], n["string"]))
    return notes


def load_model_events(path: Path) -> List[Dict]:
    events: List[Dict] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].startswith("#") or parts[0] == "Time":
            continue
        try:
            onset = float(parts[0])
        except ValueError:
            continue
        if parts[1] not in STRING_NAMES:
            continue
        index = STRING_NAMES.index(parts[1])
        fret = int(parts[2])
        events.append(
            {
                "time": onset,
                "duration": float(parts[3].rstrip("s")),
                "string": index + 1,
                "fret": fret,
                "pitch": _STRING_PITCHES[index] + fret,
                "techniques": parts[4] if len(parts) > 4 else "",
            }
        )
    events.sort(key=lambda e: (e["time"], e["string"]))
    return events


def align_pitch_sequences(
    reference: Sequence[Dict], hypothesis: Sequence[Dict]
) -> List[Tuple[int, int]]:
    n, m = len(reference), len(hypothesis)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if reference[i]["pitch"] == hypothesis[j]["pitch"]:
                table[i][j] = table[i + 1][j + 1] + 1
            else:
                table[i][j] = max(table[i + 1][j], table[i][j + 1])

    pairs: List[Tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if reference[i]["pitch"] == hypothesis[j]["pitch"]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def match_onsets(
    reference: Sequence[Dict],
    hypothesis: Sequence[Dict],
    tolerance: float = DEFAULT_ONSET_TOLERANCE,
    offset: float = 0.0,
) -> List[Tuple[int, int]]:
    used: set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for i, ref in enumerate(reference):
        target = ref["time"] + offset
        best, best_distance = None, tolerance
        for j, hyp in enumerate(hypothesis):
            if j in used or hyp["pitch"] != ref["pitch"]:
                continue
            distance = abs(hyp["time"] - target)
            if distance <= best_distance:
                best, best_distance = j, distance
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


def _prf(matched: int, n_reference: int, n_hypothesis: int) -> Dict[str, float]:
    precision = matched / max(n_hypothesis, 1)
    recall = matched / max(n_reference, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "matched": matched}


def pitch_correct(events: Sequence[Dict], reference: Sequence[Dict]) -> List[Dict]:
    by_pitch: Dict[int, Counter] = defaultdict(Counter)
    for note in reference:
        by_pitch[note["pitch"]][note["string"]] += 1

    corrected: List[Dict] = []
    for event in events:
        new = dict(event)
        if event["pitch"] in by_pitch:
            string = by_pitch[event["pitch"]].most_common(1)[0][0]
            fret = event["pitch"] - _STRING_PITCHES[string - 1]
            if 0 <= fret <= 22:
                new["string"] = string
                new["fret"] = fret
        corrected.append(new)
    return corrected


def format_tab_text(events: Sequence[Dict], header: str) -> str:
    lines = [
        f"# {header}",
        f"{'Time':>8s}  {'String':>6s}  {'Fret':>4s}  {'Duration':>8s}  Techniques",
        "-" * 52,
    ]
    for event in events:
        lines.append(
            f"{event['time']:8.2f}  {STRING_NAMES[event['string'] - 1]:>6s}  "
            f"{event['fret']:4d}  {event.get('duration', 0.0):8.2f}s  "
            f"{event.get('techniques', '')}"
        )
    return "\n".join(lines) + "\n"


def run_case_study(
    hand_path: Path,
    model_path: Path,
    bpm: float = DEFAULT_BPM,
    tolerance: float = DEFAULT_ONSET_TOLERANCE,
    corrected_output: Optional[Path] = None,
) -> Dict:
    reference = load_hand_transcription(hand_path, bpm=bpm)
    predicted = load_model_events(model_path)
    if not reference or not predicted:
        raise ValueError(
            "both the hand transcription and the model output must contain notes"
        )

    # The transcription starts at bar one; the recording has a lead-in.
    offset = predicted[0]["time"] - reference[0]["time"]

    pitch_pairs = align_pitch_sequences(reference, predicted)
    pitch_metrics = _prf(len(pitch_pairs), len(reference), len(predicted))

    onset_pairs = match_onsets(reference, predicted, tolerance, offset)
    onset_metrics = _prf(len(onset_pairs), len(reference), len(predicted))

    correct_string = sum(
        1 for i, j in pitch_pairs if reference[i]["string"] == predicted[j]["string"]
    )
    string_accuracy = correct_string / max(len(pitch_pairs), 1)
    high_e = sum(1 for e in predicted if e["string"] == 1) / len(predicted)

    predicted_distribution = Counter(e["string"] for e in predicted)
    reference_distribution = Counter(n["string"] for n in reference)

    if corrected_output is not None:
        corrected = pitch_correct(predicted, reference)
        corrected_output.write_text(
            format_tab_text(
                corrected,
                "Pitch-corrected: model pitches placed on the strings the player used",
            )
        )

    results = {
        "reference_notes": len(reference),
        "predicted_notes": len(predicted),
        "lead_in_offset_s": round(offset, 3),
        "onset_tolerance_s": tolerance,
        "bpm": bpm,
        "pitch_timing_independent": pitch_metrics,
        "note_onset_aligned": onset_metrics,
        "correct_string_given_correct_pitch": string_accuracy,
        "predicted_share_on_high_e": high_e,
        "predicted_notes_per_string": {
            STRING_NAMES[s - 1]: predicted_distribution.get(s, 0) for s in range(1, 7)
        },
        "reference_notes_per_string": {
            STRING_NAMES[s - 1]: reference_distribution.get(s, 0) for s in range(1, 7)
        },
    }
    return results


def print_report(results: Dict) -> None:
    pitch = results["pitch_timing_independent"]
    onset = results["note_onset_aligned"]
    width = 64
    print("=" * width)
    print("Improvisation Case Study — model output vs. hand transcription")
    print("=" * width)
    print(
        f"Notes: {results['predicted_notes']} predicted / "
        f"{results['reference_notes']} transcribed"
    )
    print(f"Lead-in offset: {results['lead_in_offset_s']:.2f}s\n")

    print("-- Pitch content (timing-independent, LCS-aligned) --")
    print(f"  Precision : {pitch['precision']:.4f}")
    print(f"  Recall    : {pitch['recall']:.4f}")
    print(f"  F1        : {pitch['f1']:.4f}")

    print(
        f"\n-- Note level (onset-aligned, "
        f"±{results['onset_tolerance_s'] * 1000:.0f} ms) --"
    )
    print(f"  Precision : {onset['precision']:.4f}")
    print(f"  Recall    : {onset['recall']:.4f}")
    print(f"  F1        : {onset['f1']:.4f}")

    print("\n-- String placement --")
    print(
        f"  Correct string | correct pitch : "
        f"{results['correct_string_given_correct_pitch']:.4f}"
    )
    print(
        f"  Predicted notes on high E      : {results['predicted_share_on_high_e']:.4f}"
    )
    print(f"  Predicted per string : {results['predicted_notes_per_string']}")
    print(f"  Played    per string : {results['reference_notes_per_string']}")
    print("=" * width)


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "case_study"
    parser = argparse.ArgumentParser(
        description="Score the improvisation case study against its hand transcription"
    )
    parser.add_argument(
        "--hand",
        type=str,
        default=str(root / "hand_transcription.gp5"),
        help="Guitar Pro file holding the player's transcription",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(root / "model_output_raw_tab.txt"),
        help="Event table produced by whisplab.predict",
    )
    parser.add_argument("--bpm", type=float, default=DEFAULT_BPM)
    parser.add_argument(
        "--onset-tolerance",
        type=float,
        default=DEFAULT_ONSET_TOLERANCE,
        help="Onset matching tolerance in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--write-corrected",
        type=str,
        default=None,
        help="Also write the pitch-corrected tablature to this path",
    )
    parser.add_argument(
        "--json", type=str, default=None, help="Write the metrics to this JSON file"
    )
    args = parser.parse_args()

    results = run_case_study(
        Path(args.hand),
        Path(args.model),
        bpm=args.bpm,
        tolerance=args.onset_tolerance,
        corrected_output=Path(args.write_corrected) if args.write_corrected else None,
    )
    print_report(results)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nMetrics written to {args.json}")


if __name__ == "__main__":
    main()
