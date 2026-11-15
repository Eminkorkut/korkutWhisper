"""Public reusable API; calls are serialized for safe model ownership."""

from __future__ import annotations
import time
import math
import warnings
from importlib.metadata import version
from pathlib import Path
from collections.abc import Iterable, Sequence
from types import TracebackType
from typing import Any
import numpy as np
from numpy.typing import NDArray
from .audio import decode_file, prepare_array, inspect_audio, inspect_dataset
from .backend.ctranslate import CTranslateBackend, GLOBAL_LOCK, model_identity, seed_for, elapsed
from .config import WhisperConfig, InferenceConfig, GateConfig, AudioConfig, RuntimeConfig, KorkutConfig
from .exceptions import (
    ConfigurationError,
    ModelCompatibilityError,
    InferenceError,
    InvalidAudioError,
    KorkutWhisperError,
)
from .gates.g2 import G2Gate
from .profiles import GateProfile, load_gate_profile
from .strategies import Strategy, Route
from .types import (
    TranscriptionSegment,
    TranscriptionResult,
    TimingResult,
    ModelInfo,
    BatchItem,
    GateResult,
    AudioInspection,
    DatasetInspection,
)


# High-level thread-serialized client for decoder-guided CTranslate2 Whisper transcription
class KorkutWhisper:
    """
    Public interface for loading a local CTranslate2 Whisper model and performing
    decoder-guided two-stage ('g2') or fixed-window ('fixed_1500', 'fixed_3000')
    speech-to-text transcription.

    Attributes:
        config (KorkutConfig): Validated composite configuration object.
        _closed (bool): Tracks whether `close()` has been called on this instance.
        _backend (CTranslateBackend | None): Underlying CTranslate2 / faster-whisper backend instance.
        _gate (G2Gate): Frozen G2 risk gate evaluator used when `strategy == 'g2'`.
        _strategy (Strategy): Window routing policy engine.
        _validated (bool): True if the model files, runtime versions, and decoding parameters match
            the verified Experiment 05/06 calibration environment.
        _fingerprint (str): Sampled SHA-256 fingerprint of the local `model.bin` file.
    """

    # Initialize configuration, verify model compatibility, and load the CTranslate2 backend
    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = 'cuda',
        compute_type: str = 'int8',
        strategy: str = 'g2',
        language: str = 'tr',
        task: str = 'transcribe',
        device_index: int = 0,
        cpu_threads: int = 0,
        num_workers: int = 1,
        beam_size: int = 5,
        allow_unverified: bool = False,
        seed: int = 42,
        warmup: bool = False,
        config: KorkutConfig | None = None,
    ) -> None:
        """
        Construct a `KorkutWhisper` instance from either explicit keyword arguments or a pre-built
        `KorkutConfig`, verify whether the model and runtime match the calibrated G2 profile,
        initialize the underlying `CTranslateBackend` under `GLOBAL_LOCK`, and optionally execute
        a synthetic warm-up pass so subsequent calls do not pay first-run CUDA initialization latency.

        Args:
            model_path (str | Path | None): Path to the local CTranslate2 model directory (required if `config` is None).
            device (str): Execution device ('cuda', 'cpu', or 'auto'). Defaults to 'cuda'.
            compute_type (str): Quantization/precision format (e.g., 'int8'). Defaults to 'int8'.
            strategy (str): Frame routing strategy ('g2', 'fixed_1500', or 'fixed_3000'). Defaults to 'g2'.
            language (str): Target language code. Defaults to 'tr'.
            task (str): Whisper task ('transcribe' or 'translate'). Defaults to 'transcribe'.
            device_index (int): GPU device index when using CUDA. Defaults to 0.
            cpu_threads (int): Number of CPU threads (0 for automatic). Defaults to 0.
            num_workers (int): Number of CTranslate2 worker threads. Defaults to 1.
            beam_size (int): Beam search decoding width. Defaults to 5.
            allow_unverified (bool): Permit running 'g2' on unverified models/runtimes with a warning. Defaults to False.
            seed (int): Base integer seed for deterministic CTranslate2 RNG seeding. Defaults to 42.
            warmup (bool): When True, automatically runs `self.warmup()` after loading the backend. Defaults to False.
            config (KorkutConfig | None): Optional pre-constructed `KorkutConfig` overriding individual arguments.

        Variables:
            fingerprint (str): SHA-256 fingerprint returned by `model_identity()` for `model.bin`.
            valid (bool): Whether the local model files match the verified SHA-256 hashes and byte size.
            w (WhisperConfig): Shorthand reference to `config.whisper`.

        Raises:
            ConfigurationError: If `model_path` is missing when `config` is None, or `config` is not a `KorkutConfig`.
            ModelCompatibilityError: If `strategy == 'g2'`, the setup is unverified, and `allow_unverified` is False.
            ModelLoadError: If the local CTranslate2 model files are missing or fail to load.
        """
        fingerprint: str
        valid: bool
        w: WhisperConfig

        # Build KorkutConfig from arguments if no explicit config object was supplied
        if config is None:
            if model_path is None:
                raise ConfigurationError('model_path is required')
            config = KorkutConfig(
                WhisperConfig(
                    str(model_path),
                    device,
                    compute_type,
                    device_index,
                    cpu_threads,
                    num_workers,
                    beam_size,
                    language,
                    task,
                ),
                InferenceConfig(strategy, seed),
                GateConfig(allow_unverified=allow_unverified),
                AudioConfig(),
                RuntimeConfig(warmup=warmup),
            )
        elif not isinstance(config, KorkutConfig):
            raise ConfigurationError('config must be KorkutConfig')

        # Initialize internal state, load JSON gate profile, and verify its SHA-256 hash
        self.config: KorkutConfig = config
        self._closed: bool = False
        self._warmed_up: bool = False
        self._backend: CTranslateBackend | None = None
        gate_profile: GateProfile = load_gate_profile(
            config.gate.profile,
            config.gate.profile_path,
        )
        self._gate: G2Gate = G2Gate(gate_profile)
        self._strategy: Strategy = Strategy(config.inference.strategy)

        # Check model file hashes, JSON gate profile SHA-256 hash, and runtime environment
        valid, fingerprint = model_identity(config.whisper.model_path)
        w = config.whisper
        self._validated: bool = bool(
            valid
            and gate_profile.verified
            and w.compute_type == 'int8'
            and w.device == 'cuda'
            and w.language == 'tr'
            and w.task == 'transcribe'
            and w.beam_size == 5
            and version('ctranslate2') == '4.8.3'
        )

        # Enforce or warn when applying the G2 gate to an unverified model or runtime configuration
        if config.inference.strategy == 'g2' and not self._validated and not config.gate.allow_unverified:
            raise ModelCompatibilityError(
                'G2 profile is unverified for this model/device/decoding configuration; '
                'explicitly set allow_unverified=True for experimental use'
            )
        if config.inference.strategy == 'g2' and not self._validated:
            warnings.warn(
                'G2 is being applied to an unverified model or runtime configuration; results are experimental',
                RuntimeWarning,
                stacklevel=2,
            )

        self._fingerprint: str = fingerprint

        # Load the CTranslate2 backend under the global lock and optionally run warm-up
        with GLOBAL_LOCK:
            self._backend = CTranslateBackend(w)
            if self.config.runtime.warmup:
                self.warmup()

    # Verify that the KorkutWhisper instance has not been closed before running operations
    def _ensure_open(self) -> None:
        """
        Check that the instance is still open and has an active backend.

        Raises:
            InferenceError: If `close()` has been called or `self._backend` is None.
        """
        if self._closed or self._backend is None:
            raise InferenceError('KorkutWhisper instance is closed')

    # Pre-initialize CTranslate2 and CUDA kernels using a synthetic silent waveform
    def warmup(self, duration_seconds: float = 1.0, *, include_retry_window: bool = False) -> float:
        """
        Execute a synthetic silent transcription pass to pre-initialize CTranslate2,
        cuBLAS, and feature-extraction kernels so that subsequent real transcriptions
        have accurate `TimingResult` and `RTF` measurements without first-run latency spikes.

        Args:
            duration_seconds (float): Length of the synthetic 16 kHz silent waveform in seconds
                (must satisfy `0.1 <= duration_seconds <= 15.0`). Defaults to 1.0.
            include_retry_window (bool): When True and the initial route uses 1500 frames,
                also executes a second warm-up pass at 3000 frames to warm up both window shapes.
                Defaults to False.

        Variables:
            start (int): Nanosecond timestamp marking the start of the warm-up operation.
            num_samples (int): Number of 16 kHz float32 samples in the synthetic silent waveform.
            wave (NDArray[np.float32]): Zero-filled 1-D float32 waveform passed to the backend.
            mel (NDArray[Any]): Extracted Mel spectrogram for the silent waveform.
            actual (int): Number of Mel content frames (`mel.shape[-1] - 1`).
            route (Route): Routing decision determining the initial encoder window size.

        Returns:
            float: Total elapsed wall-clock time spent warming up the model, in milliseconds.

        Raises:
            ConfigurationError: If `duration_seconds` is non-finite or outside `[0.1, 15.0]`.
            InferenceError: If the instance is closed or backend execution fails.
        """
        start: int
        num_samples: int
        wave: NDArray[np.float32]
        mel: NDArray[Any]
        actual: int
        route: Route

        if (
            not isinstance(duration_seconds, (int, float))
            or isinstance(duration_seconds, bool)
            or not math.isfinite(float(duration_seconds))
            or float(duration_seconds) < 0.1
            or float(duration_seconds) > 15.0
        ):
            raise ConfigurationError('warmup duration_seconds must be a finite number between 0.1 and 15.0')

        with GLOBAL_LOCK:
            self._ensure_open()
            start = time.perf_counter_ns()
            num_samples = int(round(float(duration_seconds) * 16000))
            wave = np.zeros(num_samples, dtype=np.float32)

            # Extract Mel features and run the primary encoder/decoder window
            mel = self._backend.extract(wave)
            actual = int(mel.shape[-1]) - 1
            route = self._strategy.route(actual)
            self._backend.run(
                wave,
                mel,
                route.initial_frames,
                seed_for(self.config.inference.seed, '__warmup__', route.initial_frames),
            )

            # Optionally warm up the 3000-frame stage-2 retry window as well
            if include_retry_window and route.initial_frames != 3000:
                self._backend.run(
                    wave,
                    mel,
                    3000,
                    seed_for(self.config.inference.seed, '__warmup__', 3000),
                )

            self._warmed_up = True
            return elapsed(start)

    # Pre-check a single audio file or array on CPU without running GPU encoder/decoder inference
    def inspect_audio(
        self,
        audio: str | Path | NDArray[Any],
        *,
        sample_rate: int = 16000,
        audio_id: str | None = None,
        raise_on_error: bool = False,
    ) -> AudioInspection:
        """
        Inspect a single audio file or NumPy array on CPU to verify that it decodes cleanly
        and satisfies the `<= 1500` Mel-frame short-form limit before running GPU inference.

        Args:
            audio (str | Path | NDArray[Any]): Path to an audio file or a numeric NumPy audio array.
            sample_rate (int): Sampling rate in Hz when `audio` is a NumPy array. Defaults to 16000.
            audio_id (str | None): Optional identifier override.
            raise_on_error (bool): When True, raises `InvalidAudioError` or `UnsupportedAudioLengthError`
                on validation failure instead of returning `AudioInspection(valid=False, ...)`.

        Returns:
            AudioInspection: Detailed pre-inference inspection result for the audio input.
        """
        return inspect_audio(
            audio,
            sample_rate=sample_rate,
            audio_id=audio_id,
            raise_on_error=raise_on_error,
        )

    # Pre-check a directory or iterable of audio inputs on CPU without running GPU inference
    def inspect_dataset(
        self,
        dataset: str | Path | Iterable[str | Path | NDArray[Any]],
        *,
        pattern: str | Sequence[str] | None = None,
        recursive: bool = False,
        sample_rate: int = 16000,
    ) -> DatasetInspection:
        """
        Inspect a directory of audio files or an iterable of audio inputs on CPU without invoking
        the GPU Whisper encoder/decoder, returning aggregate counts, Mel-frame statistics, and
        per-item eligibility results.

        Args:
            dataset (str | Path | Iterable[str | Path | NDArray[Any]]): Directory path or iterable of audios.
            pattern (str | Sequence[str] | None): Glob pattern(s) when scanning a directory.
            recursive (bool): Whether to scan subdirectories recursively. Defaults to False.
            sample_rate (int): Sampling rate in Hz for any NumPy arrays in `dataset`. Defaults to 16000.

        Returns:
            DatasetInspection: Aggregated dataset inspection report and item list.
        """
        return inspect_dataset(
            dataset,
            pattern=pattern,
            recursive=recursive,
            sample_rate=sample_rate,
        )

    # Transcribe an audio file path or pre-loaded 16 kHz NumPy waveform
    def transcribe(self, audio: str | Path | NDArray[Any], *, audio_id: str | None = None) -> TranscriptionResult:
        """
        Decode an audio file (via PyAV) or validate a 16 kHz NumPy waveform array, then
        run the configured transcription pipeline.

        Args:
            audio (str | Path | NDArray[Any]): Path to an audio file or a 16 kHz NumPy audio array.
            audio_id (str | None): Optional identifier used for deterministic RNG seeding and metadata
                (defaults to the file stem for paths or `'array'` for NumPy arrays).

        Variables:
            decode_start (int): Nanosecond timestamp marking the start of audio decoding/preparation.
            name (str): Resolved `audio_id` string passed into the execution pipeline.
            start (int): Nanosecond timestamp marking the overall start of the `transcribe()` call.
            wave (NDArray[np.float32]): Decoded and normalized 16 kHz mono float32 waveform.

        Returns:
            TranscriptionResult: Complete transcription output including text, segments, gate decision, and timings.

        Raises:
            InferenceError: If the instance is closed or backend execution fails.
            InvalidAudioError: If `audio` is neither a path nor a valid NumPy array, or decoding fails.
            UnsupportedAudioLengthError: If the audio length in Mel frames is unsupported by the active strategy.
        """
        decode_start: int
        name: str
        start: int
        wave: NDArray[np.float32]

        self._ensure_open()
        start = time.perf_counter_ns()
        decode_start = time.perf_counter_ns()

        # Decode from file path or validate in-memory 16 kHz NumPy array
        if isinstance(audio, (str, Path)):
            wave = decode_file(audio)
            name = str(audio_id) if audio_id is not None else Path(audio).stem
        elif isinstance(audio, np.ndarray):
            wave = prepare_array(audio, 16000)
            name = str(audio_id) if audio_id is not None else 'array'
        else:
            raise InvalidAudioError('audio must be a path or NumPy array')

        return self._execute(wave, name, start, elapsed(decode_start))

    # Transcribe an in-memory audio array with arbitrary sampling rate
    def transcribe_array(
        self,
        samples: Any,
        sample_rate: int = 16000,
        *,
        audio_id: str = 'array',
    ) -> TranscriptionResult:
        """
        Validate, downmix, normalize, and resample an in-memory audio array from `sample_rate`
        to 16 kHz mono float32, then execute the transcription pipeline.

        Args:
            samples (Any): 1-D (mono) or 2-D (samples x channels) numeric audio array.
            sample_rate (int): Sampling rate of `samples` in Hz. Defaults to 16000.
            audio_id (str): Identifier used for deterministic RNG seeding and metadata. Defaults to `'array'`.

        Variables:
            prep (int): Nanosecond timestamp marking the start of array preparation and resampling.
            start (int): Nanosecond timestamp marking the overall start of `transcribe_array()`.
            wave (NDArray[np.float32]): Prepared 16 kHz mono float32 waveform.

        Returns:
            TranscriptionResult: Complete transcription output including text, segments, gate decision, and timings.
        """
        prep: int
        start: int
        wave: NDArray[np.float32]

        self._ensure_open()
        start = time.perf_counter_ns()
        prep = time.perf_counter_ns()
        wave = prepare_array(samples, sample_rate)
        return self._execute(wave, str(audio_id), start, elapsed(prep))

    # Core serialized execution pipeline: Mel extraction, routing, stage-1 inference, G2 gate, and optional stage-2 retry
    def _execute(
        self,
        wave: NDArray[np.float32],
        audio_id: str,
        start: int,
        decode_ms: float,
    ) -> TranscriptionResult:
        """
        Execute feature extraction, frame-window routing, stage-1 decoding, G2 risk evaluation,
        and conditional stage-2 (3000-frame) retry while holding `GLOBAL_LOCK`.

        Args:
            wave (NDArray[np.float32]): Normalized 16 kHz mono float32 audio waveform.
            audio_id (str): Identifier string used to derive deterministic CTranslate2 seeds.
            start (int): Nanosecond start timestamp of the outer transcription request.
            decode_ms (float): Elapsed milliseconds spent decoding or preparing `wave`.

        Variables:
            actual (int): Number of actual Mel spectrogram content frames (`mel.shape[-1] - 1`).
            duration (float): Audio duration in seconds (`len(wave) / 16000`).
            final (list[Any]): Decoded segment list from the final selected stage (stage 1 or stage 2).
            first (dict[str, Any]): Stage-1 encoder/decoder timing and shape telemetry dictionary.
            gate_ms (float): Elapsed milliseconds spent evaluating the G2 gate (0.0 if bypassed).
            gate_result (GateResult | None): Output of `G2Gate.evaluate()`, or None if gate was not used.
            initial (list[Any]): Decoded segment list produced by stage-1 inference.
            mel (NDArray[Any]): Pre-extracted 2-D log-Mel spectrogram reused across stages.
            pre (int): Nanosecond timestamp marking the start of Mel feature extraction.
            pre_ms (float): Elapsed milliseconds spent extracting the Mel spectrogram.
            retry (bool): True if a stage-2 (3000-frame) retry pass was executed.
            route (Route): Routing decision specifying `initial_frames` and `use_gate`.
            second (dict[str, Any] | None): Stage-2 telemetry dictionary if retried, otherwise None.
            segments (list[TranscriptionSegment]): Sanitized public `TranscriptionSegment` objects.
            tick (int): Nanosecond timestamp marking the start of G2 gate evaluation.
            timing (TimingResult): Aggregated wall-clock timing breakdown and Real-Time Factor.
            total (float): Total end-to-end wall-clock time in milliseconds since `start`.

        Returns:
            TranscriptionResult: Final structured transcription result with telemetry and metadata.
        """
        actual: int
        duration: float
        final: list[Any]
        first: dict[str, Any]
        gate_ms: float
        gate_result: GateResult | None
        initial: list[Any]
        mel: NDArray[Any]
        pre: int
        pre_ms: float
        retry: bool
        route: Route
        second: dict[str, Any] | None
        segments: list[TranscriptionSegment]
        tick: int
        timing: TimingResult
        total: float

        with GLOBAL_LOCK:
            self._ensure_open()
            duration = len(wave) / 16000

            # Extract Mel spectrogram once and determine the routing policy from its frame count.
            # Note: Strategy.route() strictly enforces actual <= 1500 Mel content frames (~15s)
            # and raises UnsupportedAudioLengthError for longer inputs. Future work will introduce
            # a pre-routing chunking layer for long-form audio (>1500 Mel frames).
            pre = time.perf_counter_ns()
            mel = self._backend.extract(wave)
            pre_ms = elapsed(pre)
            actual = int(mel.shape[-1]) - 1
            route = self._strategy.route(actual)

            # Run stage-1 inference at route.initial_frames (1500 or 3000)
            initial, first = self._backend.run(
                wave,
                mel,
                route.initial_frames,
                seed_for(self.config.inference.seed, audio_id, route.initial_frames),
            )

            gate_result = None
            gate_ms = 0.0
            second = None
            final = initial

            # Evaluate the G2 gate when eligible and trigger a 3000-frame stage-2 retry if required
            if route.use_gate:
                tick = time.perf_counter_ns()
                gate_result = self._gate.evaluate(initial)
                gate_ms = elapsed(tick)
                if gate_result.retry_required:
                    final, second = self._backend.run(
                        wave,
                        mel,
                        3000,
                        seed_for(self.config.inference.seed, audio_id, 3000),
                    )

            total = elapsed(start)
            retry = second is not None

            # Helper to convert finite numeric metrics to float and sanitize NaN/Inf/None to None
            def finite(value: Any) -> float | None:
                """
                Return `float(value)` if `value` is not None and finite, otherwise return None.
                """
                return float(value) if value is not None and math.isfinite(float(value)) else None

            # Convert raw faster-whisper segments into public TranscriptionSegment dataclasses
            segments = [
                TranscriptionSegment(
                    text=s.text.strip(),
                    start=None,
                    end=None,
                    avg_logprob=finite(getattr(s, 'avg_logprob', None)),
                    compression_ratio=finite(getattr(s, 'compression_ratio', None)),
                    tokens=[int(t) for t in getattr(s, 'tokens', [])]
                    if getattr(s, 'tokens', None) is not None
                    else None,
                )
                for s in final
            ]

            # Build stage-by-stage timing summary and Real-Time Factor (RTF)
            timing = TimingResult(
                decode_ms,
                pre_ms,
                first['encoder_ms'],
                first['decoder_ms'],
                second['encoder_ms'] if second else None,
                second['decoder_ms'] if second else None,
                gate_ms,
                total,
                total / (duration * 1000),
            )

            return TranscriptionResult(
                ' '.join(s.text for s in segments if s.text),
                self.config.whisper.language,
                duration,
                self.config.inference.strategy,
                retry,
                2 if retry else 1,
                segments,
                gate_result,
                timing,
                {
                    'routing_reason': route.reason,
                    'actual_mel_frames': actual,
                    'encoder_input_frames': [
                        s[-1]
                        for s in first['encoder_shapes'] + (second['encoder_shapes'] if second else [])
                    ],
                    'encoder_calls': len(first['encoder_shapes'])
                    + (len(second['encoder_shapes']) if second else 0),
                    'audio_id': audio_id,
                    'mel_reused_for_retry': retry,
                    'timestamps_available': False,
                    'validated_profile': self._validated,
                    'warmed_up': self._warmed_up,
                    'timing_scope': 'host wall time; no explicit GPU synchronization; stage timings overlap total',
                    'first_stage_text': ' '.join(s.text.strip() for s in initial if s.text.strip()),
                },
            )

    # Sequentially transcribe a batch of audio files or arrays, capturing per-item errors
    def transcribe_many(
        self,
        audios: Iterable[str | Path | NDArray[Any]],
        *,
        max_concurrency: int = 1,
    ) -> list[BatchItem]:
        """
        Transcribe an iterable of audio file paths or NumPy arrays sequentially, wrapping each
        outcome or `KorkutWhisperError` into a `BatchItem` so a single bad input does not abort the batch.

        Args:
            audios (Iterable[str | Path | NDArray[Any]]): Sequence/iterable of audio paths or 16 kHz arrays.
            max_concurrency (int): Must be 1 (sequential execution). Defaults to 1.

        Variables:
            audio (str | Path | NDArray[Any]): Current audio item being transcribed.
            i (int): Zero-based index of the current item in `audios`.
            results (list[BatchItem]): Accumulated list of `BatchItem` results.

        Returns:
            list[BatchItem]: Ordered list of batch outcomes corresponding to `audios`.

        Raises:
            InferenceError: If the `KorkutWhisper` instance is closed.
            ConfigurationError: If `max_concurrency` is not 1.
        """
        audio: str | Path | NDArray[Any]
        i: int
        results: list[BatchItem]

        self._ensure_open()
        if max_concurrency != 1:
            raise ConfigurationError('only sequential max_concurrency=1 is supported')

        results = []
        for i, audio in enumerate(audios):
            try:
                results.append(BatchItem(i, self.transcribe(audio)))
            except KorkutWhisperError as exc:
                results.append(BatchItem(i, None, type(exc).__name__, str(exc)))
        return results

    # Return metadata and verification status for the currently loaded model
    def get_model_info(self) -> ModelInfo:
        """
        Construct and return a `ModelInfo` summary describing the loaded model path,
        device, compute type, Mel bin count, gate profile, fingerprint, and verification status.

        Variables:
            w (WhisperConfig): Shorthand reference to `self.config.whisper`.

        Returns:
            ModelInfo: Populated model metadata dataclass.

        Raises:
            InferenceError: If the instance has been closed.
        """
        w: WhisperConfig
        self._ensure_open()
        w = self.config.whisper
        return ModelInfo(
            str(Path(w.model_path).resolve()),
            'Whisper',
            w.compute_type,
            w.device,
            self._backend.mel_bins,
            w.language,
            w.task,
            '0.1.0',
            self.config.gate.profile if self.config.inference.strategy == 'g2' else None,
            self._fingerprint,
            self._validated,
        )

    # Return the active KorkutConfig serialized as a plain dictionary
    def get_config(self) -> dict[str, Any]:
        """
        Return a nested dictionary representation of `self.config`.

        Returns:
            dict[str, Any]: Serialized configuration dictionary.
        """
        return self.config.to_dict()

    # Return the active G2Gate calibration parameters as a plain dictionary
    def get_gate_profile(self) -> dict[str, Any]:
        """
        Return the frozen calibration weights, statistics, and threshold of the G2 risk gate.

        Returns:
            dict[str, Any]: Dictionary of G2 gate parameters from `self._gate.to_dict()`.
        """
        return self._gate.to_dict()

    # Close the backend and release underlying model resources under GLOBAL_LOCK
    def close(self) -> None:
        """
        Idempotently close the underlying `CTranslateBackend` and mark this instance as closed.
        """
        with GLOBAL_LOCK:
            if not self._closed:
                if self._backend is not None:
                    self._backend.close()
                self._backend = None
                self._closed = True

    # Enter the runtime context manager, ensuring the instance is open
    def __enter__(self) -> KorkutWhisper:
        """
        Verify the instance is open and return `self` for `with KorkutWhisper(...) as model:` usage.

        Returns:
            KorkutWhisper: This active instance.
        """
        self._ensure_open()
        return self

    # Exit the runtime context manager and close model resources
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """
        Close the instance when leaving a `with` block without suppressing exceptions.

        Args:
            exc_type (type[BaseException] | None): Exception class if an error was raised in the block.
            exc_value (BaseException | None): Exception instance if an error was raised.
            traceback (TracebackType | None): Traceback object if an error was raised.

        Returns:
            bool: Always False so any exception inside the `with` block propagates normally.
        """
        self.close()
        return False

