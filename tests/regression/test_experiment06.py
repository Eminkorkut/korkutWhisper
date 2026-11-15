"""Opt-in regression: compares archived GPU outputs; never changes them.

KORKUTWHISPER_REGRESSION=1 pytest tests/regression -v
KORKUTWHISPER_REGRESSION_ALL=1 requests all 3473 (expensive).
"""

import csv
import os
import unicodedata
from pathlib import Path
import pytest
from korkutwhisper import KorkutWhisper

# Repository paths for archived Experiment 06 predictions, model weights, and test audio
ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / 'results/06-frozen-g2-gate-test-20261115_152900_d21a/gate_predictions.csv'
MODEL = ROOT / 'large-V3-turbo-exp2-LoRA-ct2-int8'
AUDIO = ROOT / 'data/ISSAI_TSC_218/Test'


# Normalize Turkish text to lowercase NFC without punctuation or symbols for WER evaluation
def normalize(text):
    """
    Normalize a Turkish string by applying Unicode NFC normalization, mapping Turkish
    uppercase `'İ'` -> `'i'` and `'I'` -> `'ı'`, lowercasing, stripping combining dot above,
    and replacing all punctuation/symbol characters with spaces.

    Args:
        text (str): Input reference or hypothesis string.

    Returns:
        str: Whitespace-normalized lowercase text string.
    """
    text = (
        unicodedata.normalize('NFC', text)
        .replace('İ', 'i')
        .replace('I', 'ı')
        .lower()
        .replace('\u0307', '')
    )
    return ' '.join(
        ''.join(' ' if unicodedata.category(c).startswith(('P', 'S')) else c for c in text).split()
    )


# Compute word-level Levenshtein edit distance between reference and prediction strings
def word_errors(reference, prediction):
    """
    Calculate the number of word-level insertions, deletions, and substitutions
    between `reference` and `prediction` after applying `normalize()`.

    Args:
        reference (str): Ground-truth reference transcription text.
        prediction (str): Model-predicted transcription text.

    Variables:
        a (list[str]): Tokenized words from the normalized reference string.
        b (list[str]): Tokenized words from the normalized prediction string.
        prev (list[int]): Previous dynamic programming row of edit distances.
        cur (list[int]): Current dynamic programming row being computed.

    Returns:
        int: Minimum word-level Levenshtein edit distance.
    """
    a = normalize(reference).split()
    b = normalize(prediction).split()
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# Opt-in GPU regression test comparing live transcription outputs against Experiment 06 archive
@pytest.mark.skipif(os.getenv('KORKUTWHISPER_REGRESSION') != '1', reason='opt-in GPU/model/data regression')
def test_archived_experiment06():
    """
    Run `KorkutWhisper` on CUDA against archived Experiment 06 test samples and verify
    that stage-1 text, gate features, risk score, retry decision, final text, encoder
    frame windows, and total word errors match the archived CSV records.
    """
    import ctranslate2

    if ctranslate2.get_cuda_device_count() == 0:
        pytest.skip('CUDA device unavailable')
    if not ARCHIVE.is_file() or not MODEL.is_dir() or not AUDIO.is_dir():
        pytest.skip('Experiment 06 archive, model or Test data unavailable')

    with ARCHIVE.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))

    # Select a representative subset unless KORKUTWHISPER_REGRESSION_ALL=1 is requested
    if os.getenv('KORKUTWHISPER_REGRESSION_ALL') != '1':
        rows = [
            next(r for r in rows if r['retry'] == 'False'),
            next(r for r in rows if r['retry'] == 'True'),
        ] + rows[:18]
        rows = list({r['audio_id']: r for r in rows}.values())

    expected_errors = actual_errors = 0
    with KorkutWhisper(str(MODEL), device='cuda', compute_type='int8') as asr:
        for row in rows:
            aid = row['audio_id']
            result = asr.transcribe(AUDIO / f'{aid}.wav', audio_id=aid)
            assert result.metadata['first_stage_text'] == row['transcript_1500'], aid
            assert result.gate.features['avg_logprob'] == pytest.approx(
                float(row['avg_logprob']), abs=1e-6, rel=0
            ), aid
            assert result.gate.features['compression_ratio'] == pytest.approx(
                float(row['compression_ratio']), abs=1e-6, rel=0
            ), aid
            assert result.gate.risk_score == pytest.approx(
                float(row['risk_score']), abs=1e-6, rel=0
            ), aid
            assert result.gate.threshold == float(row['risk_threshold']), aid
            assert result.retry_performed == (row['retry'] == 'True'), aid
            assert result.text == row['final_transcript'], aid
            assert result.metadata['encoder_input_frames'] == (
                [1500, 3000] if result.retry_performed else [1500]
            ), aid
            ref = (AUDIO / f'{aid}.txt').read_text(encoding='utf-8-sig')
            actual_errors += word_errors(ref, result.text)
            expected_errors += int(row['word_errors'])

    assert actual_errors == expected_errors


# Verify G2Gate risk scores and decisions across all 3,473 archived rows without GPU inference
@pytest.mark.skipif(not ARCHIVE.is_file(), reason='Experiment 06 archive unavailable')
def test_all_archived_g2_scores_and_decisions_without_inference():
    """
    Feed all 3,473 archived segment feature pairs (`avg_logprob`, `compression_ratio`)
    into `G2Gate.evaluate()` and assert 1e-12 agreement on `risk_score` and exact
    agreement on `retry_required`.
    """
    from types import SimpleNamespace
    from korkutwhisper.gates.g2 import G2Gate

    with ARCHIVE.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3473
    for row in rows:
        segment = SimpleNamespace(
            avg_logprob=float(row['avg_logprob']) if row['avg_logprob'] else None,
            compression_ratio=float(row['compression_ratio']) if row['compression_ratio'] else None,
        )
        result = G2Gate().evaluate([segment])
        assert result.risk_score == pytest.approx(
            float(row['risk_score']), abs=1e-12, rel=0
        ), row['audio_id']
        assert result.retry_required == (row['retry'] == 'True'), row['audio_id']

