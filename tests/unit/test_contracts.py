"""Unit tests verifying G2 gate equations, audio array validation, routing rules, and config serialization."""

import json
from types import SimpleNamespace
import numpy as np
import pytest
from korkutwhisper import (
    KorkutConfig,
    WhisperConfig,
    InferenceConfig,
    GateConfig,
    RuntimeConfig,
    ConfigurationError,
    InvalidAudioError,
    UnsupportedAudioLengthError,
    TranscriptionResult,
    TranscriptionSegment,
    GateResult,
    TimingResult,
    BatchItem,
    format_batch_csv,
    save_batch_csv,
    save_batch_jsonl,
    inspect_audio,
    inspect_dataset,
    VERIFIED_PROFILE_HASHES,
    load_gate_profile,
    cli_main,
)
from korkutwhisper.audio import prepare_array
from korkutwhisper.gates import g2
from korkutwhisper.strategies import Strategy


# Verify the G2 linear risk equation, threshold boundary decision, and median imputation for missing features
def test_g2_exact_equation_missing_and_boundary(monkeypatch):
    """
    Test that `G2Gate.evaluate()` computes the exact linear combination of standardized
    `avg_logprob` and `compression_ratio`, triggers retry when `risk_score >= THRESHOLD`,
    and imputes missing/NaN features using calibration medians while remaining JSON-serializable.

    Args:
        monkeypatch: Pytest fixture used to temporarily override `g2.THRESHOLD`.

    Variables:
        gate (g2.G2Gate): Instance of the G2 decoder-risk gate under test.
        s (SimpleNamespace): Mock decoded segment with valid `avg_logprob` and `compression_ratio`.
        r (GateResult): Evaluation result for the valid segment `s`.
        expected (float): Manually computed reference risk score from module constants.
        a (GateResult): Evaluation result for a segment with `None` and `NaN` features.
    """
    gate = g2.G2Gate()
    s = SimpleNamespace(avg_logprob=-.2, compression_ratio=1.2)
    r = gate.evaluate([s])
    expected = (
        g2.LOGPROB_WEIGHT * ((-.2 - g2.LOGPROB_MEAN) / g2.LOGPROB_STD)
        + g2.COMPRESSION_WEIGHT * ((1.2 - g2.COMPRESSION_MEAN) / g2.COMPRESSION_STD)
    )
    assert r.risk_score == expected

    # Verify boundary condition: risk_score == THRESHOLD must require a retry
    monkeypatch.setattr(g2, 'THRESHOLD', r.risk_score)
    assert gate.evaluate([s]).retry_required is True

    # Verify median imputation when features are None or NaN
    a = gate.evaluate([SimpleNamespace(avg_logprob=None, compression_ratio=float('nan'))])
    assert a.imputed_features == ['avg_logprob', 'compression_ratio'] and np.isfinite(a.risk_score)
    json.dumps(a.to_dict(), allow_nan=False)


# Verify audio array resampling, stereo-to-mono downmixing, and rejection of invalid arrays
def test_audio_validation_and_resampling():
    """
    Test that `prepare_array()` resamples 8 kHz input to 16 kHz, downmixes 2-channel input
    to 1-D mono, and raises `InvalidAudioError` for empty arrays, NaNs, >8 channels, or
    unnormalized floating-point samples.
    """
    assert len(prepare_array(np.zeros(8000), 8000)) == 16000
    assert prepare_array(np.zeros((5, 2)), 16000).shape == (5,)
    for x in ([], [float('nan')], [[1] * 20]):
        with pytest.raises(InvalidAudioError):
            prepare_array(x, 16000)
    with pytest.raises(InvalidAudioError):
        prepare_array([1.2], 16000)


# Verify strategy frame routing and strict 1500-Mel-frame upper bound enforcement
def test_routes_no_truncation():
    """
    Test that `Strategy.route()` selects 1500 or 3000 Mel encoder frames for valid
    inputs (`1 <= actual <= 1500`) across `'g2'`, `'fixed_1500'`, and `'fixed_3000'`,
    and raises `UnsupportedAudioLengthError` whenever input frames are `<= 0` or `> 1500`.
    """
    assert Strategy('g2').route(1000).initial_frames == 1500
    assert Strategy('g2').route(1500).initial_frames == 1500
    assert Strategy('g2').route(1500).use_gate is True
    assert Strategy('fixed_1500').route(1500).initial_frames == 1500
    assert Strategy('fixed_3000').route(10).initial_frames == 3000

    for strategy_name in ('g2', 'fixed_1500', 'fixed_3000'):
        with pytest.raises(UnsupportedAudioLengthError):
            Strategy(strategy_name).route(0)
        with pytest.raises(UnsupportedAudioLengthError):
            Strategy(strategy_name).route(1501)
        with pytest.raises(UnsupportedAudioLengthError):
            Strategy(strategy_name).route(3001)


# Verify JSON configuration save/load roundtrip and concurrency validation
def test_config_roundtrip(tmp_path):
    """
    Test that `KorkutConfig.save_json()` and `KorkutConfig.load_json()` preserve
    all configuration fields identically, and that `max_concurrency != 1` is rejected.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Variables:
        c (KorkutConfig): Sample composite configuration instance.
        p (Path): Temporary file path for saving and reloading JSON config.
    """
    c = KorkutConfig(
        WhisperConfig('/tmp/model'),
        InferenceConfig(),
        GateConfig(),
        runtime=RuntimeConfig(),
    )
    p = tmp_path / 'config.json'
    c.save_json(p)
    assert KorkutConfig.load_json(p) == c
    with pytest.raises(ConfigurationError):
        RuntimeConfig(max_concurrency=2)


# Verify all single-result and batch output formatters (JSON, CSV, Report, TXT, SRT, VTT, JSONL)
def test_output_formats_and_batch_exports(tmp_path):
    """
    Test that `TranscriptionResult` and `BatchItem` serialize cleanly across all
    supported output formats (`text`, `json`, `csv`, `report`, `srt`, `vtt`, and `.jsonl`).

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Variables:
        result (TranscriptionResult): Sample transcription result with G2 gate and timing metrics.
        batch (list[BatchItem]): Sample batch containing one successful item and one failed item.
    """
    result = TranscriptionResult(
        text='merhaba dünya',
        language='tr',
        duration=2.5,
        strategy='g2',
        retry_performed=False,
        final_stage=1,
        segments=[
            TranscriptionSegment(
                text='merhaba dünya',
                start=None,
                end=None,
                avg_logprob=-0.03,
                compression_ratio=0.95,
                tokens=[10, 20],
            )
        ],
        gate=GateResult(
            name='decoder_based_b2',
            risk_score=-0.04,
            threshold=0.078,
            retry_required=False,
            features={'avg_logprob': -0.03, 'compression_ratio': 0.95},
        ),
        timing=TimingResult(
            audio_decode_ms=3.0,
            preprocessing_ms=4.0,
            encoder_stage1_ms=15.0,
            decoder_stage1_ms=25.0,
            encoder_stage2_ms=None,
            decoder_stage2_ms=None,
            gate_ms=0.1,
            total_ms=47.1,
            rtf=0.01884,
        ),
        metadata={
            'audio_id': '3487',
            'routing_reason': 'g2_eligible',
            'encoder_input_frames': [1500],
            'first_stage_text': 'merhaba dünya',
        },
    )

    # Plain text and human-readable diagnostic report
    assert result.to_text() == 'merhaba dünya'
    assert '=== KorkutWhisper Transcription Report (3487) ===' in result.to_report()
    assert 'merhaba dünya' in result.to_report()

    # JSON and CSV strings
    assert json.loads(result.to_json())['text'] == 'merhaba dünya'
    csv_text = result.to_csv()
    assert 'audio_id,text,first_stage_text' in csv_text
    assert '3487,merhaba dünya,merhaba dünya' in csv_text

    # Subtitle formats (SRT and WebVTT)
    assert '00:00:00,000 --> 00:00:02,500\nmerhaba dünya' in result.to_srt()
    assert result.to_vtt().startswith('WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nmerhaba dünya')

    # Extension-inferred file saving via result.save()
    for ext in ('txt', 'json', 'csv', 'srt', 'vtt'):
        out_file = result.save(tmp_path / f'sample.{ext}')
        assert out_file.is_file() and out_file.read_text(encoding='utf-8').strip()

    with pytest.raises(ConfigurationError):
        result.format('xml')

    # Batch CSV and JSONL export with mixed success and error items
    batch = [
        BatchItem(index=0, result=result),
        BatchItem(index=1, result=None, error_type='InvalidAudioError', error_message='audio is empty'),
    ]
    batch_csv = format_batch_csv(batch)
    assert '0,ok,3487,merhaba dünya' in batch_csv
    assert '1,error,' in batch_csv and 'InvalidAudioError,audio is empty' in batch_csv

    csv_path = save_batch_csv(batch, tmp_path / 'batch.csv')
    jsonl_path = save_batch_jsonl(batch, tmp_path / 'batch.jsonl')
    assert csv_path.is_file()
    assert len(jsonl_path.read_text(encoding='utf-8').strip().splitlines()) == 2


# Verify CPU-only single-audio and dataset pre-inference inspection
def test_inspect_audio_and_dataset(tmp_path):
    """
    Test `inspect_audio()` and `inspect_dataset()` on eligible short audio (`<=1500` frames),
    long audio (`>1500` frames), and invalid/corrupt inputs without loading a GPU model.
    """
    # 1. Single valid short audio array (5 seconds = 500 Mel frames)
    short_info = inspect_audio(np.zeros(5 * 16000, dtype=np.float32), audio_id='short_clip')
    assert short_info.valid is True
    assert short_info.status == 'eligible'
    assert short_info.actual_mel_frames == 500
    assert '=== Audio Inspection (short_clip) ===' in short_info.to_report()

    # 2. Single long audio array exceeding 1500 Mel frames (16 seconds = 1600 Mel frames)
    long_info = inspect_audio(np.zeros(16 * 16000, dtype=np.float32), audio_id='long_clip')
    assert long_info.valid is False
    assert long_info.status == 'exceeds_1500_frames'
    assert long_info.actual_mel_frames == 1600
    assert long_info.error_type == 'UnsupportedAudioLengthError'

    with pytest.raises(UnsupportedAudioLengthError):
        inspect_audio(np.zeros(16 * 16000, dtype=np.float32), raise_on_error=True)

    # 3. Single invalid audio array (contains NaN)
    bad_info = inspect_audio(np.array([np.nan], dtype=np.float32), audio_id='bad_clip')
    assert bad_info.valid is False
    assert bad_info.status == 'invalid_audio'
    assert bad_info.error_type == 'InvalidAudioError'

    with pytest.raises(InvalidAudioError):
        inspect_audio(np.array([np.nan], dtype=np.float32), raise_on_error=True)

    # 4. Dataset inspection across mixed items
    ds = inspect_dataset(
        [
            np.zeros(5 * 16000, dtype=np.float32),
            np.zeros(16 * 16000, dtype=np.float32),
            np.array([np.nan], dtype=np.float32),
        ]
    )
    assert ds.total_count == 3
    assert ds.eligible_count == 1
    assert ds.exceeds_1500_count == 1
    assert ds.invalid_count == 1
    assert len(ds.eligible_items()) == 1
    assert len(ds.rejected_items()) == 2
    assert '=== KorkutWhisper Dataset Inspection Report ===' in ds.to_report()
    csv_out = ds.save_csv(tmp_path / 'inspection.csv')
    assert csv_out.is_file()


# Verify JSON GateProfile loading, SHA-256 verification, and tamper detection
def test_gate_profile_json_and_sha256_verification(tmp_path):
    """
    Test that the built-in `decoder_based_b2.json` profile passes SHA-256 verification,
    while a modified/custom JSON profile is flagged as `verified=False`.
    """
    builtin = load_gate_profile('decoder_based_b2')
    assert builtin.profile == 'decoder_based_b2'
    assert builtin.verified is True
    assert builtin.sha256 == VERIFIED_PROFILE_HASHES['decoder_based_b2']

    # Write a modified profile JSON (altered threshold) and confirm verified=False
    custom_path = tmp_path / 'custom_profile.json'
    data = builtin.to_dict()
    data['threshold'] = 0.05
    custom_path.write_text(json.dumps(data, indent=2), encoding='utf-8')

    custom = load_gate_profile(profile_path=custom_path)
    assert custom.threshold == 0.05
    assert custom.verified is False
    assert custom.sha256 != VERIFIED_PROFILE_HASHES['decoder_based_b2']


# Verify CLI execution for --gate-info and error handling when --model is omitted
def test_cli_gate_info_and_validation(capsys):
    """
    Test that `cli_main(['--gate-info'])` prints the verified JSON profile with exit code 0,
    and that invoking transcription without `--model` returns exit code 1.
    """
    code = cli_main(['--gate-info'])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload['profile'] == 'decoder_based_b2'
    assert payload['verified'] is True

    err_code = cli_main(['nonexistent.wav'])
    assert err_code == 1





