from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import guitarpro

from whisplab.config import STANDARD_TUNING
from whisplab.predict import STRING_NAMES

SLOTS_PER_MEASURE = 16  # 4/4 notated on a sixteenth-note grid

# Sixteenth-note count -> (Guitar Pro duration value, dotted)
_DURATION_TABLE: Dict[int, Tuple[int, bool]] = {
    1: (16, False),  # sixteenth
    2: (8, False),  # eighth
    3: (8, True),  # dotted eighth
    4: (4, False),  # quarter
    6: (4, True),  # dotted quarter
    8: (2, False),  # half
    12: (2, True),  # dotted half
    16: (1, False),  # whole
}


def parse_tab_txt(path: Path) -> List[Dict]:
    events: List[Dict] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].startswith("#") or parts[0] == "Time":
            continue
        try:
            onset = float(parts[0])
        except ValueError:
            continue  # separator rules and other decoration
        if parts[1] not in STRING_NAMES:
            continue
        events.append(
            {
                "time": onset,
                "duration": float(parts[3].rstrip("s")),
                "string": STRING_NAMES.index(parts[1]) + 1,
                "fret": int(parts[2]),
            }
        )
    events.sort(key=lambda e: (e["time"], e["string"]))
    return events


def _split_duration(slots: int) -> List[Tuple[int, bool]]:
    out: List[Tuple[int, bool]] = []
    remaining = slots
    while remaining > 0:
        for length in sorted(_DURATION_TABLE, reverse=True):
            if length <= remaining:
                out.append(_DURATION_TABLE[length])
                remaining -= length
                break
        else:  # pragma: no cover - _DURATION_TABLE contains 1, so always matched
            break
    return out


def _quantise(
    events: Sequence[Dict], grid: float, monophonic: bool = True
) -> Dict[int, List[Dict]]:
    slots: Dict[int, List[Dict]] = {}
    for ev in events:
        slot = round(ev["time"] / grid)
        bucket = slots.setdefault(slot, [])
        if any(n["string"] == ev["string"] for n in bucket):
            continue
        length = max(1, round(ev["duration"] / grid))
        bucket.append({"string": ev["string"], "fret": ev["fret"], "slots": length})

    if monophonic:
        for slot, bucket in slots.items():
            slots[slot] = [min(bucket, key=lambda n: n["string"])]
    return slots


def events_to_song(
    events: Sequence[Dict],
    bpm: int = 100,
    tuning: Sequence[int] = STANDARD_TUNING,
    title: str = "whisplab transcription",
) -> guitarpro.Song:
    song = guitarpro.Song()
    song.title = title
    song.tempo = bpm
    song.tracks = []

    track = guitarpro.Track(song)
    track.name = "Guitar"
    # PyGuitarPro orders strings high -> low; STANDARD_TUNING is low -> high.
    track.strings = [
        guitarpro.GuitarString(number=i + 1, value=v)
        for i, v in enumerate(reversed(list(tuning)))
    ]
    track.measures = []

    grid = (60.0 / bpm) / 4.0  # seconds per sixteenth
    slots = _quantise(events, grid)
    if not slots:
        raise ValueError("no note events to export")

    # The final note is truncated at the barline rather than spilling into an
    # extra measure, so the span is set by the last onset.
    n_measures = max(1, -(-(max(slots) + 1) // SLOTS_PER_MEASURE))  # ceil

    for m_index in range(n_measures):
        header = guitarpro.MeasureHeader()
        header.number = m_index + 1
        measure = guitarpro.Measure(track, header)
        measure.voices = [guitarpro.Voice(measure) for _ in range(2)]
        for voice in measure.voices:
            voice.beats = []

        voice = measure.voices[0]
        base = m_index * SLOTS_PER_MEASURE
        cursor = 0
        while cursor < SLOTS_PER_MEASURE:
            notes = slots.get(base + cursor)
            if notes is None:
                # Gather the rest until the next occupied slot in this measure.
                span = 1
                while (
                    cursor + span < SLOTS_PER_MEASURE
                    and slots.get(base + cursor + span) is None
                ):
                    span += 1
                for value, dotted in _split_duration(span):
                    beat = guitarpro.Beat(voice)
                    beat.duration = guitarpro.Duration(value=value, isDotted=dotted)
                    beat.notes = []
                    voice.beats.append(beat)
                cursor += span
                continue

            # A sounded beat lasts until the note ends or the next slot begins,
            # whichever comes first, and never crosses the barline.
            next_onset = SLOTS_PER_MEASURE
            for ahead in range(cursor + 1, SLOTS_PER_MEASURE):
                if slots.get(base + ahead) is not None:
                    next_onset = ahead
                    break
            span = max(n["slots"] for n in notes)
            span = max(1, min(span, next_onset - cursor))

            for value, dotted in _split_duration(span)[:1]:
                beat = guitarpro.Beat(voice)
                beat.duration = guitarpro.Duration(value=value, isDotted=dotted)
                beat.notes = []
                for n in notes:
                    note = guitarpro.Note(beat)
                    note.string = n["string"]
                    note.value = n["fret"]
                    note.type = guitarpro.NoteType.normal
                    beat.notes.append(note)
                voice.beats.append(beat)
            cursor += span

        track.measures.append(measure)

    song.tracks.append(track)
    song.measureHeaders = [m.header for m in track.measures]
    return song


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a whisplab tablature event list into a .gp5 file"
    )
    parser.add_argument("tab_txt", type=str, help="A *_tab.txt file from predict")
    parser.add_argument(
        "--bpm", type=int, default=100, help="Tempo of the notated grid (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination .gp5 (default: alongside the input)",
    )
    args = parser.parse_args()

    tab_path = Path(args.tab_txt)
    out_path = (
        Path(args.output)
        if args.output
        else tab_path.with_name(tab_path.stem.replace("_tab", "") + ".gp5")
    )

    events = parse_tab_txt(tab_path)
    song = events_to_song(events, bpm=args.bpm, title=tab_path.stem)
    guitarpro.write(song, str(out_path))
    print(f"{len(events)} events → {out_path}")


if __name__ == "__main__":
    main()
