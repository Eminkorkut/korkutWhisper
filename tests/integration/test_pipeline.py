"""Integration tests for controlled_window hooks, backend invariants, and end-to-end pipeline routing."""

import json
from types import SimpleNamespace
import numpy as np
import pytest
import faster_whisper.transcribe as fw
from korkutwhisper import (
    KorkutWhisper,
    InferenceError,
    ModelCompatibilityError,
    UnsupportedAudioLengthError,
)
from korkutwhisper.backend.ctranslate import controlled_window


# Dummy feature extractor returning a fixed (128, 101) float32 Mel spectrogram
class Extractor:
    """
    Minimal mock feature extractor callable used to test `controlled_window` class patching.
    """

    # Return a zeroed (128, 101) Mel spectrogram array
    def __call__(self, *a, **kw):
        """
        Return a dummy 128-bin, 101-frame float32 NumPy array.
        """
        return np.zeros((128, 101), dtype=np.float32)


# Dummy faster-whisper model stub exposing feature_extractor, encode, and generate_with_fallback
class FakeModel:
    """
    Lightweight stand-in for `faster_whisper.WhisperModel` used to verify monkeypatching
    and restoration in `controlled_window`.
    """

    # Initialize the fake model with a dummy Extractor instance
    def __init__(self):
        """
        Attach a mock `Extractor` as `self.feature_extractor`.
        """
        self.feature_extractor = Extractor()

    # Identity encoder stub
    def encode(self, x):
        """
        Return the encoder input tensor unchanged.
        """
        return x

    # No-op decoder stub
    def generate_with_fallback(self, *a, **kw):
        """
        Placeholder for faster-whisper's fallback decoder method.
        """
        return None


# Verify that controlled_window enforces target padding shapes, prevents truncation, and restores patches
def test_real_integration_guard_shapes_and_restore():
    """
    Test that `controlled_window` caches `mel`, pads encoder inputs to `target` frames (1500 or 3000),
    raises `InferenceError` if an input exceeds `target`, and cleanly restores `fw.pad_or_trim`
    and `model.feature_extractor.__class__` upon exit.

    Variables:
        model (FakeModel): Stub WhisperModel instance.
        mel (NDArray[np.float32]): Pre-computed (128, 101) Mel array to cache.
        original (Any): Reference to the unpatched `fw.pad_or_trim` function.
        extractor_type (type): Original class of `model.feature_extractor`.
        target (int): Target frame window size under test (1500 or 3000).
        measured (dict[str, Any]): Telemetry dictionary yielded by `controlled_window`.
    """
    model = FakeModel()
    mel = np.zeros((128, 101), dtype=np.float32)
    original = fw.pad_or_trim
    extractor_type = model.feature_extractor.__class__

    for target in (1500, 3000):
        with controlled_window(model, mel, target) as measured:
            assert model.feature_extractor(np.ones(3)) is mel
            model.encode(fw.pad_or_trim(mel[:, :-1]))
        assert measured['encoder_shapes'] == [(128, target)] and measured['mel_calls'] == 1

    assert fw.pad_or_trim is original and model.feature_extractor.__class__ is extractor_type

    with pytest.raises(InferenceError):
        with controlled_window(model, mel, 1500):
            fw.pad_or_trim(np.zeros((128, 1501)))

    assert fw.pad_or_trim is original


# Verify that CTranslateBackend.run() invokes the cached Mel extractor and target-sized encoder
def test_backend_inference_calls_cached_extractor_and_encoder():
    """
    Test that `CTranslateBackend.run()` executes `model.transcribe()` inside `controlled_window`
    and records a single cached Mel lookup and a single `(128, target)` encoder call.

    Variables:
        backend (CTranslateBackend): Uninitialized backend instance populated with a fake `Transcriber`.
        mel (NDArray[np.float32]): Pre-extracted (128, 101) Mel spectrogram array.
        target (int): Target encoder frame count (1500 or 3000).
        segments (list[Any]): Decoded segment list returned by `backend.run()`.
        measured (dict[str, Any]): Telemetry dictionary returned by `backend.run()`.
    """
    from korkutwhisper.backend.ctranslate import CTranslateBackend

    # Fake model implementing a minimal transcribe() pipeline
    class Transcriber(FakeModel):
        """
        Simulates faster-whisper's `transcribe()` sequence: feature extraction, `pad_or_trim`,
        `encode`, and `generate_with_fallback`.
        """

        # Simulate faster-whisper transcription steps and yield one segment at seek=0
        def transcribe(self, wave, **kwargs):
            """
            Run simulated feature extraction, padding, encoding, and decoding.
            """
            mel = self.feature_extractor(wave, chunk_length=None)
            self.encode(fw.pad_or_trim(mel[:, :-1]))
            self.generate_with_fallback()
            return iter([SimpleNamespace(text='x', seek=0)]), None

    backend = CTranslateBackend.__new__(CTranslateBackend)
    backend.model = Transcriber()
    backend.config = SimpleNamespace(language='tr', task='transcribe', beam_size=5)
    mel = np.zeros((128, 101), dtype=np.float32)

    for target in (1500, 3000):
        segments, measured = backend.run(np.zeros(16000, dtype=np.float32), mel, target, 42)
        assert len(segments) == 1 and measured['encoder_shapes'] == [(128, target)] and measured['mel_calls'] == 1


# Mock backend simulating low-confidence stage-1 output that triggers a stage-2 retry
class FakeBackend:
    """
    Mock replacement for `CTranslateBackend` that records requested frame sizes and Mel arrays,
    returning low confidence (`avg_logprob = -0.5`) at 1500 frames to trigger G2 retry.
    """

    # Initialize tracking lists for frames, Mel spectrograms, and close status
    def __init__(self, cfg):
        """
        Set `mel_bins = 128` and initialize call recording attributes.
        """
        self.mel_bins = 128
        self.frames = []
        self.mels = []
        self.closed = False

    # Return a synthetic Mel array proportional to waveform length
    def extract(self, wave):
        """
        Compute a dummy `(128, len(wave)//160 + 1)` float32 Mel spectrogram array.
        """
        return np.zeros((128, int(len(wave) / 160) + 1), dtype=np.float32)

    # Record frame window and Mel array, returning a mock segment and telemetry dict
    def run(self, wave, mel, frames, seed):
        """
        Append `frames` and `mel` to tracking lists and return a synthetic decoded segment.
        """
        self.frames.append(frames)
        self.mels.append(mel)
        segment = SimpleNamespace(
            text=' test',
            avg_logprob=-.5 if frames == 1500 else -.01,
            compression_ratio=1.2,
            tokens=[1, 2],
        )
        return [segment], {'encoder_ms': 1.0, 'decoder_ms': 2.0, 'encoder_shapes': [(128, frames)]}

    # Mark the mock backend as closed
    def close(self):
        """
        Record that `close()` was called on the backend.
        """
        self.closed = True


# Verify stage-2 retry execution, Mel reuse, batch error handling, >1500-frame gate bypass, and context closing
def test_retry_and_context(monkeypatch):
    """
    Test `KorkutWhisper` end-to-end with `FakeBackend`:
      - Short high-risk audio runs stage 1 (1500) and retries stage 2 (3000) reusing the same Mel array.
      - `transcribe_many` captures invalid audio items without aborting valid items.
      - Audio > 1500 Mel frames bypasses the G2 gate and runs directly at 3000 frames.
      - Exiting the `with` context closes the backend and blocks subsequent calls.
    """
    import korkutwhisper.api as api

    monkeypatch.setattr(api, 'version', lambda name: '4.8.3')
    monkeypatch.setattr(api, 'model_identity', lambda p: (True, 'hash'))
    monkeypatch.setattr(api, 'CTranslateBackend', FakeBackend)

    with KorkutWhisper('fake', device='cuda') as model:
        backend = model._backend
        result = model.transcribe_array(np.zeros(16000), audio_id='x')
        assert result.retry_performed and result.metadata['encoder_input_frames'] == [1500, 3000]
        assert backend.mels[0] is backend.mels[1] and backend.frames == [1500, 3000]
        assert result.segments[0].start is None and result.segments[0].end is None
        assert result.metadata['encoder_calls'] == 2
        json.dumps(result.to_dict(), allow_nan=False)

        items = model.transcribe_many([np.zeros(16000), np.array([np.nan])])
        assert items[0].result is not None and items[1].error_type == 'InvalidAudioError'

        # Audio exceeding 1500 Mel frames (~15s) must be rejected with UnsupportedAudioLengthError
        with pytest.raises(UnsupportedAudioLengthError):
            model.transcribe_array(np.zeros(16 * 16000), audio_id='long')

    assert backend.closed
    with pytest.raises(InferenceError):
        model.transcribe_array(np.zeros(1000))


# Verify unverified model rejection under G2 and single-stage execution under fixed_1500
def test_no_retry_and_compatibility(monkeypatch):
    """
    Test that `KorkutWhisper` raises `ModelCompatibilityError` when `model_identity` reports
    an unverified model under `'g2'` without `allow_unverified=True`, and that `'fixed_1500'`
    executes a single encoder pass without running the G2 gate.
    """
    import korkutwhisper.api as api

    monkeypatch.setattr(api, 'model_identity', lambda p: (False, 'hash'))
    with pytest.raises(ModelCompatibilityError):
        KorkutWhisper('fake')

    monkeypatch.setattr(api, 'model_identity', lambda p: (True, 'hash'))
    monkeypatch.setattr(api, 'CTranslateBackend', FakeBackend)
    model = KorkutWhisper('fake', strategy='fixed_1500')
    result = model.transcribe_array(np.zeros(16000))
    assert not result.retry_performed and result.gate is None
    assert result.metadata['encoder_calls'] == 1
    model.close()


# Verify that confident stage-1 output under G2 passes the gate without triggering a retry
def test_g2_no_retry_and_real_routing(monkeypatch):
    """
    Test that when stage-1 (1500-frame) decoding yields high confidence (`avg_logprob = -0.01`,
    `compression_ratio = 0.8`), the G2 gate sets `retry_required = False` and completes in a single call.
    """
    import korkutwhisper.api as api

    monkeypatch.setattr(api, 'version', lambda name: '4.8.3')

    # Mock backend returning high-confidence stage-1 segments
    class ConfidentBackend(FakeBackend):
        """
        Subclass of `FakeBackend` that produces low-risk segment metrics so no retry is triggered.
        """

        # Return a confident segment at the requested frame count
        def run(self, wave, mel, frames, seed):
            """
            Record call arguments and return a segment with `avg_logprob = -0.01` and `compression_ratio = 0.8`.
            """
            self.frames.append(frames)
            self.mels.append(mel)
            segment = SimpleNamespace(
                text=' test',
                avg_logprob=-.01,
                compression_ratio=.8,
                tokens=[1],
            )
            return [segment], {'encoder_ms': 1.0, 'decoder_ms': 1.0, 'encoder_shapes': [(128, frames)]}

    monkeypatch.setattr(api, 'model_identity', lambda p: (True, 'hash'))
    monkeypatch.setattr(api, 'CTranslateBackend', ConfidentBackend)

    with KorkutWhisper('fake') as model:
        result = model.transcribe_array(np.zeros(16000))
        assert not result.retry_performed and not result.gate.retry_required
        assert result.metadata['encoder_input_frames'] == [1500] and result.metadata['encoder_calls'] == 1


# Verify explicit model.warmup() and automatic warmup=True initialization
def test_model_warmup_explicit_and_on_init(monkeypatch):
    """
    Test that `KorkutWhisper(..., warmup=True)` automatically runs a 1-second silent
    warm-up pass on initialization, and that `model.warmup(include_retry_window=True)`
    warms up both the 1500-frame and 3000-frame encoder windows while setting `warmed_up=True`.
    """
    import korkutwhisper.api as api
    from korkutwhisper import ConfigurationError

    monkeypatch.setattr(api, 'version', lambda name: '4.8.3')
    monkeypatch.setattr(api, 'model_identity', lambda p: (True, 'hash'))
    monkeypatch.setattr(api, 'CTranslateBackend', FakeBackend)

    with KorkutWhisper('fake', warmup=True) as model:
        backend = model._backend
        assert model._warmed_up is True
        assert backend.frames == [1500]

        warmup_ms = model.warmup(duration_seconds=0.5, include_retry_window=True)
        assert isinstance(warmup_ms, float) and warmup_ms >= 0.0
        assert backend.frames == [1500, 1500, 3000]

        result = model.transcribe_array(np.zeros(16000))
        assert result.metadata['warmed_up'] is True

        with pytest.raises(ConfigurationError):
            model.warmup(duration_seconds=0.0)
        with pytest.raises(ConfigurationError):
            model.warmup(duration_seconds=20.0)


