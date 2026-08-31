# whisplab

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22212907.svg)](https://doi.org/10.5281/zenodo.22212907)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)

**Automatic Transcription of Guitar Improvisations from Audio Recordings.**

whisplab turns a recording of an electric guitar into tablature. Where standard
transcription recovers _what pitch_ was sounded, tablature is prescriptive — it
says _how the piece was played_, committing every note to a specific string and
fret. whisplab predicts that directly, jointly estimating per-string fret
labels, note onsets, and eleven playing techniques at frame level.

This is the reference implementation for the bachelor's thesis \_Automatic Transcription of Guitar Improvisations from Audio Recordings. (Johannes Kepler
University Linz, 2026).

```
$ python -m whisplab.predict --audio solo.wav --midi

# Predicted Guitar Tablature
# Total duration: 26.0s
# 120 note events detected

    Time  String  Fret  Duration  Techniques
----------------------------------------------------
    2.39       e     8      2.37s
    4.76       e    10      0.19s
    4.95       e    12      0.19s
    ...
    7.01       e    17      0.21s  b
```

---

## Results

Evaluated on the held-out GOAT test split — seven full-length recordings,
each processed in one pass.

| Metric                                       | Value           |
| -------------------------------------------- | --------------- |
| **Note F1**                                  | **0.6984**      |
| Note precision / recall                      | 0.5629 / 0.9200 |
| Mean frame accuracy                          | 0.7721          |
| Mean active-note accuracy                    | 0.8305          |
| Tablature accuracy (all six strings correct) | 0.3356          |
| Onset F1 (event-based, ±46 ms)               | 0.5312          |
| Technique micro-F1                           | 0.0893          |

Recall is high and precision lower: the model finds nearly every sounded note
but fires in more string-frames than the ground truth contains. Tablature
accuracy is strict — one wrong string out of six invalidates the whole frame.

Two caveats belong next to these numbers rather than in a footnote. The test
split holds only seven recordings, and per-recording note F1 ranges from 0.67
to 0.98, so the pooled figures carry real variance. And technique detection
essentially does not work: six of the eleven techniques have no positive frames
in the test set at all, and of the five that occur, only the dead note is
detected. See [Limitations](#limitations).

## The improvisation case study

GOAT cannot measure string placement. Its string labels are derived
deterministically from pitch, so every target places a note on the highest
feasible string — the model is never scored against the string a performer
actually chose.

To measure that, a 26-second improvised solo was recorded, transcribed by hand,
and passed through the model. It is the only direct measurement of tablature
quality in this work, and it is unflattering in an informative way:

| Measure                                | Value |
| -------------------------------------- | ----- |
| Pitch content, timing-independent F1   | 0.70  |
| Note level, onset-aligned (±100 ms) F1 | 0.25  |
| Correct string given correct pitch     | 27 %  |
| Predicted notes on the high-E string   | 76 %  |

The model recovers **what** was played but not **where** on the neck. Notes the
player took on the B, G, and D strings are systematically re-assigned upward to
the high E — precisely the behaviour the pitch-derived training labels predict.
Re-fingering the model's pitches onto the strings the player used, without
changing a single pitch, recovers a tablature close to the original:

```
True (player's transcription)
e|---------10--12------10--12--13--12--10--12------10--12--17--|
B|-13--13----------13--------------------------13-------------|

Model output (raw)
e|-8--10--12--8--10--12--13--12--10--12--10--12--13--17--10--5-|
B|------------------------------------------------------------|

Model output (pitch-corrected onto the player's strings)
e|-----10--12------10--12--13--12--10--12--10--12--13--17--10--|
B|-13----------13---------------------------------------------|
```

The audio, the hand transcription, both model transcriptions, and the metrics
are in [`case_study/`](case_study/), and archived on Zenodo at
[10.5281/zenodo.22212907](https://doi.org/10.5281/zenodo.22212907).
**All figures reproduce from the artifacts in this repository** —
`tests/test_case_study.py` asserts each published number.

## Installation

Requires Python 3.14.

```bash
git clone https://github.com/MaximilianJPaul/whisplab.git
cd whisplab
poetry install
```

Or with pip:

```bash
pip install -e .
```

Training and evaluation use CUDA where available, then Apple MPS, then CPU.

### Pretrained checkpoint

`best_model.pt` is 176 MB — past GitHub's file limit — and is not currently
distributed, so `train.py` is the only way to produce one here. It is available
from the author on request. Place it at `checkpoints/best_model.pt`, or point
any command at it with `--checkpoint`.

Note that this affects what the case study reproduces from a clean clone:
scoring runs from the committed artifacts, but regenerating the transcription
from the audio needs the checkpoint.

### Dataset

Training needs the GOAT dataset, which is not redistributed here. See
[`data/README.md`](data/README.md) for how to obtain it and the expected
directory layout.

## Usage

Transcribe audio:

```bash
python -m whisplab.predict --audio solo.wav --output-dir predictions --midi
```

This writes `predictions/solo_tab.txt` — one row per detected note event, with
unquantised onset, duration, string, fret and detected techniques — and, with
`--midi`, a MIDI rendering.

Render a transcription as a Guitar Pro file:

```bash
python -m whisplab.export_gp predictions/solo_tab.txt --bpm 100 --output solo.gp5
```

Onsets are quantised to a sixteenth-note grid at the given tempo, so the
`.gp5` loses timing detail the `_tab.txt` keeps. Use the text output whenever
exact timing matters.

Train:

```bash
python -m whisplab.train --max-epochs 120 --batch-size 8
```

Evaluate on the GOAT test split:

```bash
python -m whisplab.evaluate --checkpoint checkpoints/best_model.pt
```

Add `--tune` to fit onset and technique thresholds on the test set. That is an
oracle upper bound, not held-out performance — the reported results all use a
fixed logit threshold of zero.

Score the case study:

```bash
python -m whisplab.case_study --json case_study/metrics.json
```

Each command is also installed as a console script
(`whisplab-predict`, `whisplab-train`, and so on).

## How it works

```
waveform
  └─ CQT (192 bins, 24 per octave, fmin = C1)      nnAudio, on-GPU
     └─ log scaling → SpecAugment (training only)
        └─ multi-scale CNN front-end                3×3 and 5×5 paths,
           │                                        SE channel attention,
           │                                        residual, 4 stages
           └─ linear projection → sinusoidal positional encoding
              └─ Transformer encoder                8 layers, 8 heads,
                 │                                  d_model 256, Pre-LN
                 ├─→ 6 × fret head          (T, 24)   frets 0–22 + inactive
                 ├─→ 6 × onset head         (T, 1)
                 └─→ cross-string attention
                     └─→ 6 × technique head (T, 11)
```

**Frame-level, per-string classification.** Every 23.2 ms frame gets one
prediction per string: a fret from 0 to 22, or `inactive`. Tablature is read
off directly, with no note-level decoding stage.

**A CQT front-end.** The Constant-Q Transform's logarithmic frequency spacing
puts a constant number of bins per semitone, so a chord's harmonic pattern
keeps the same shape wherever it sits on the neck — a translation the
convolutional front-end can exploit. Two parallel kernel sizes capture
fine-grained and broader spectro-temporal structure.

**Sinusoidal positional encoding**, not learned, so the encoder can process
full-length recordings at inference having only ever trained on 10-second
crops. (How well it does so is itself a finding — see Limitations.)

**Cross-string attention before the technique heads** lets each string's
technique prediction attend to the other five, which matters for techniques
that span strings.

**Multi-task loss.** Class-weighted focal cross-entropy for frets (silence
dominates), plus weighted BCE for onsets and techniques. Onset targets are
smeared with a triangular window over ±2 frames, since exact-frame onset
labels are far too sparse to learn from.

Training uses mixup, ±2-semitone pitch shifting with matching label
transposition, random gain, EMA weight averaging, one-cycle LR scheduling, and
gradient accumulation to an effective batch size of 32. Key hyperparameters
live in [`src/whisplab/config.py`](src/whisplab/config.py).

## Limitations

**String assignment is not learned from performance.** GOAT's string labels are
derived from pitch, placing each note on the highest feasible string, so the
model reproduces that convention rather than a player's fingering. This is the
central limitation, and the case study measures its cost: 27 % correct string
placement on genuinely out-of-distribution audio. Fixing it needs training data
with performed string annotations.

**Technique detection does not generalise.** Micro-F1 is 0.0893 on the test
set. The cause is extreme class imbalance — most techniques occupy well under
1 % of frames, and 5.9 hours of audio does not contain enough positives.
Validation micro-F1 reaches ≈0.40, but the validation split shares recordings
and players with training, so that gap is memorisation, not generalisation.
Tuning thresholds on the test set directly still lifts no class above ≈0.31.

**The validation split is sample-level, not recording-level.** Different
amplifier variants of the same recording can appear in both training and
validation, making validation an optimistic early-stopping signal. Only the
seven test recordings are strictly held out. No validation figure is reported
as a result.

**Length extrapolation is imperfect.** Training uses 10-second crops
(≈430 frames); test recordings run up to 184 seconds (≈7900 frames). Splitting
long recordings into independent 10-second chunks raises pooled note F1 from
0.698 to 0.711, so the mismatch is real but modest.

**Timing is approximate**, and frame-level metrics punish it harshly against
beat-quantised ground truth. For the practical goal — handing a guitarist a
usable first draft of their own playing — the system is best described as a
pitch transcriber with a canonical fingering: the notes are substantially
right; the fingering and exact rhythm need manual correction.

## Citing

If you use this work, please cite the thesis and the software. Machine-readable
metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders a formatted
citation from the sidebar.

The case-study data is separately citable:

> Voshchepynets, M. J. P. (2026). _Improvisation Case Study: Audio, Hand
> Transcription and Model Output for Automatic Guitar Tablature Transcription_
> (1.0.0) [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.22212907>

## Acknowledgements

Built on the **GOAT** dataset (Loth et al., 2025), the largest public
collection of real electric guitar recordings with aligned tablature and
technique annotations. Technique labels come from its DadaGP token encoding
(Sarmento et al., 2021). The CQT front-end uses
[nnAudio](https://github.com/KinWaiCheuk/nnAudio); Guitar Pro I/O uses
[PyGuitarPro](https://github.com/Perlence/PyGuitarPro).

## Licence

MIT — see [LICENSE](LICENSE). The case-study audio recording and hand
transcription are the author's own work, released under CC BY 4.0.
