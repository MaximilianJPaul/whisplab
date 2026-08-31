from pathlib import Path

import pytest

from whisplab.case_study import (
    align_pitch_sequences,
    load_hand_transcription,
    load_model_events,
    pitch_correct,
    run_case_study,
)

CASE_STUDY = Path(__file__).resolve().parents[1] / "case_study"
HAND = CASE_STUDY / "hand_transcription.gp5"
RAW = CASE_STUDY / "model_output_raw_tab.txt"

pytestmark = pytest.mark.skipif(
    not (HAND.exists() and RAW.exists()),
    reason="case study artifacts are not present",
)


@pytest.fixture(scope="module")
def results():
    return run_case_study(HAND, RAW)


def test_note_counts(results):
    assert results["reference_notes"] == 84
    assert results["predicted_notes"] == 120


def test_lead_in_offset(results):
    assert results["lead_in_offset_s"] == pytest.approx(2.39, abs=0.01)


def test_timing_independent_pitch_metrics(results):
    pitch = results["pitch_timing_independent"]
    assert pitch["matched"] == 71
    assert pitch["recall"] == pytest.approx(0.85, abs=0.005)
    assert pitch["precision"] == pytest.approx(0.59, abs=0.005)
    assert pitch["f1"] == pytest.approx(0.70, abs=0.005)


def test_onset_aligned_metrics(results):
    onset = results["note_onset_aligned"]
    assert onset["f1"] == pytest.approx(0.25, abs=0.005)


def test_string_placement_is_the_dominant_error(results):
    # Pitch is largely recovered, but only ~27% of correctly-pitched notes land
    # on the string the player used, and ~76% are pushed onto the high E.
    assert results["correct_string_given_correct_pitch"] == pytest.approx(
        0.27, abs=0.005
    )
    assert results["predicted_share_on_high_e"] == pytest.approx(0.76, abs=0.005)


def test_pitch_correction_preserves_pitch_and_spreads_strings():
    reference = load_hand_transcription(HAND)
    predicted = load_model_events(RAW)
    corrected = pitch_correct(predicted, reference)

    assert len(corrected) == len(predicted)
    # Re-fingering must never change what note sounds, only where it is played.
    for before, after in zip(predicted, corrected):
        assert before["pitch"] == after["pitch"]
        assert before["time"] == after["time"]

    # The correction is the point: notes leave the high E string.
    before_high_e = sum(1 for e in predicted if e["string"] == 1)
    after_high_e = sum(1 for e in corrected if e["string"] == 1)
    assert after_high_e < before_high_e


def test_pitch_alignment_is_order_preserving():
    reference = load_hand_transcription(HAND)
    predicted = load_model_events(RAW)
    pairs = align_pitch_sequences(reference, predicted)

    for (i0, j0), (i1, j1) in zip(pairs, pairs[1:]):
        assert i0 < i1 and j0 < j1
    for i, j in pairs:
        assert reference[i]["pitch"] == predicted[j]["pitch"]
