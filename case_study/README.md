# Improvisation Case Study — Data and Artifacts

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22212907.svg)](https://doi.org/10.5281/zenodo.22212907)

Supplementary materials for the improvisation case study in the bachelor's
thesis _Automatic Transcription of Guitar Improvisations from Audio Recordings._ (Johannes Kepler University Linz, 2026), produced with
[whisplab](https://github.com/MaximilianJPaul/whisplab).

Archived on Zenodo at
[10.5281/zenodo.22212907](https://doi.org/10.5281/zenodo.22212907).

## Why this case study exists

Every quantitative result in the thesis comes from the GOAT test split, and
GOAT cannot measure string placement. Its string labels are derived
deterministically from pitch, so every target places a note on the highest
feasible string — the model is never shown, and never scored against, the
string a performer actually chose.

This case study is the one setting in the thesis where the _performed_ string
assignments are known. A 26-second improvised electric-guitar solo was
recorded, transcribed by hand to obtain ground truth, and passed through the
trained model. It is therefore the only direct measurement of how well the
system recovers tablature rather than pitch.

## Contents

| File                                  | Description                                                                             |
| ------------------------------------- | --------------------------------------------------------------------------------------- |
| `audio/improvisation_case_study.wav`  | The source recording: 26 s improvised solo, 44.1 kHz, 16-bit stereo                     |
| `hand_transcription.gp5`              | The player's own transcription — ground truth, 84 notes                                 |
| `model_output_raw.gp5`                | The model's transcription, as predicted (90 notated notes)                              |
| `model_output_pitchcorrected.gp5`     | The model's pitches re-fingered onto the strings the player used                        |
| `model_output_raw_tab.txt`            | Unquantised model output: 120 note events with onset, duration, string, fret, technique |
| `model_output_pitchcorrected_tab.txt` | The same 120 events after re-fingering                                                  |
| `model_output_raw.mid`                | The model output as MIDI                                                                |
| `metrics.json`                        | All figures below, machine-readable                                                     |

The `.gp5` files open in Guitar Pro, TuxGuitar, and most tablature editors.
The `_tab.txt` files are the canonical output — they preserve unquantised
timing, which the notated `.gp5` versions necessarily lose.

## Results

| Measure                                | Value       |
| -------------------------------------- | ----------- |
| Pitch content, timing-independent F1   | 0.70        |
| — recall / precision                   | 0.85 / 0.59 |
| Note level, onset-aligned (±100 ms) F1 | 0.25        |
| Correct string given correct pitch     | 27 %        |
| Predicted notes on the high-E string   | 76 %        |
| Predicted / transcribed note count     | 120 / 84    |

Because the hand transcription is quantised to a 100 BPM grid while the
performance has fluid timing, a strict note-for-note comparison would be
misleading. Two complementary views are therefore reported: a
**timing-independent** comparison of the pitch sequences, aligned order-aware
by longest common subsequence, and a conventional **onset-aligned** note
comparison at a ±100 ms tolerance.

**Pitch content is largely recovered.** On a timing-independent basis the
model finds 85 % of the played pitches. The lower precision reflects
over-segmentation: it emits 120 onsets for 84 played notes, splitting
sustained notes into repeated attacks.

**String placement is the dominant error.** Of the correctly-pitched notes,
only 27 % sit on the string actually played, and 76 % of all predictions are
pushed onto the high-E string. The confusions are systematic — notes taken on
the B, G, and D strings are re-assigned upward — which is exactly what the
pitch-derived training labels predict. The model recovers _what_ was played
but not _where_ on the neck.

**Timing is approximate.** The gap between the timing-independent F1 (0.70)
and the onset-aligned F1 (0.25) quantifies the timing error. Part is genuine
model jitter; much of it is the mismatch between a beat-quantised
transcription and the performer's micro-timing.

The `model_output_pitchcorrected` files isolate these effects. Re-fingering the
model's pitches onto the strings the player used — without changing a single
pitch — recovers a tablature close to the original performance, showing that
the pitch content and the string assignment fail independently.

## Reproducing these numbers

Everything here regenerates from the audio and the trained checkpoint:

```bash
python -m whisplab.predict --audio case_study/audio/improvisation_case_study.wav --output-dir case_study --midi
```

Scoring needs no checkpoint — only the files in this directory:

```bash
python -m whisplab.case_study --json case_study/metrics.json
```

`tests/test_case_study.py` in the repository asserts each published figure
against these artifacts.

## Licence

Code and metrics: MIT. The audio recording and the hand transcription are the
author's own work, released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

Please cite the thesis and this deposit:

> Voshchepynets, M. J. P. (2026). _Improvisation Case Study: Audio, Hand
> Transcription and Model Output for Automatic Guitar Tablature Transcription_
> (1.0.0) [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.22212907>

See `CITATION.cff` in the
[repository](https://github.com/MaximilianJPaul/whisplab) for the software
citation.
