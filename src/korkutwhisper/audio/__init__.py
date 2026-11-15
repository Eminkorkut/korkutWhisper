"""Audio inputs and pre-inference inspection utilities; file decode matches the Experiment 06 PyAV path."""

from __future__ import annotations
from collections.abc import Iterable, Sequence
import io
import math
import statistics
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly
from ..exceptions import InvalidAudioError, UnsupportedAudioLengthError, ConfigurationError
from ..strategies import MAX_CONTENT_FRAMES
from ..types import AudioInspection, DatasetInspection

# Default audio file glob patterns scanned when inspect_dataset() is given a directory
DEFAULT_AUDIO_PATTERNS: tuple[str, ...] = ('*.wav', '*.flac', '*.mp3', '*.ogg', '*.m4a')


# Decode an audio file from disk into a normalized 16 kHz mono float32 waveform
def decode_file(path: str | Path) -> NDArray[np.float32]:
    """
    Decode an audio file using PyAV, resample it to 16 kHz mono 16-bit PCM,
    and normalize the samples into a float32 NumPy array in the [-1.0, 1.0] range.

    Args:
        path (str | Path): Filesystem path to the input audio file.

    Variables:
        arr (NDArray[Any]): Temporary NumPy array converted from each resampled PyAV frame.
        buf (io.BytesIO): In-memory byte buffer accumulating raw resampled PCM audio chunks.
        container (Any): Opened PyAV input container for reading audio streams.
        dtype (Any): NumPy data type of the resampled integer PCM frames (defaults to np.int16).
        frame (Any): Current decoded audio frame yielded by the PyAV stream iterator.
        item (Any): Individual resampled frame output produced by AudioResampler.
        iterator (Any): Generator yielding decoded frames from the first audio stream (audio=0).
        resampler (Any): PyAV AudioResampler configured for signed 16-bit ('s16'), mono, 16000 Hz output.
        wave (NDArray[np.float32]): Final normalized 1-D float32 waveform divided by 32768.0.

    Returns:
        NDArray[np.float32]: 1-D contiguous float32 audio waveform sampled at 16 kHz.

    Raises:
        InvalidAudioError: If the file does not exist, contains no audio stream, has corrupt
            frames, decodes to an empty buffer, or fails decoding for any other reason.
    """
    arr: NDArray[Any]
    buf: io.BytesIO
    container: Any
    dtype: Any
    frame: Any
    item: Any
    iterator: Any
    resampler: Any
    wave: NDArray[np.float32]
    import av

    # Resolve path and verify the audio file exists on disk
    path = Path(path)
    if not path.is_file():
        raise InvalidAudioError(f'audio file not found: {path}')

    # Initialize the 16 kHz mono s16 resampler and in-memory PCM buffer
    resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=16000)
    buf = io.BytesIO()
    dtype = np.int16

    try:
        # Open the media container and decode the primary audio stream frame by frame
        with av.open(str(path), mode='r') as container:
            if not container.streams.audio:
                raise InvalidAudioError('file has no audio stream')
            iterator = container.decode(audio=0)
            while True:
                try:
                    frame = next(iterator)
                except StopIteration:
                    break
                except av.error.InvalidDataError as exc:
                    raise InvalidAudioError('corrupt audio frame; refusing partial transcription') from exc

                # Clear presentation timestamp to prevent resampler discontinuity checks
                frame.pts = None
                for item in resampler.resample(frame):
                    arr = item.to_ndarray()
                    dtype = arr.dtype
                    buf.write(arr)

            # Flush any remaining buffered samples from the resampler
            for item in resampler.resample(None):
                arr = item.to_ndarray()
                dtype = arr.dtype
                buf.write(arr)

        # Convert accumulated int16 PCM bytes into normalized [-1.0, 1.0] float32 samples
        wave = np.frombuffer(buf.getbuffer(), dtype=dtype).astype(np.float32) / 32768.0
        if not wave.size:
            raise InvalidAudioError('audio is empty')
        return wave
    except InvalidAudioError:
        raise
    except Exception as exc:
        raise InvalidAudioError(f'cannot decode {path}: {exc}') from exc


# Validate, normalize, downmix, and resample an in-memory audio array to 16 kHz mono float32
def prepare_array(samples: Any, sample_rate: int) -> NDArray[np.float32]:
    """
    Validate and convert an arbitrary numeric array-like audio input into a contiguous
    1-D 16 kHz mono float32 NumPy array suitable for Whisper feature extraction.

    Args:
        samples (Any): Input 1-D (mono) or 2-D (samples x channels) numeric audio array.
        sample_rate (int): Original sampling rate of the input audio in Hz (must be a positive int).

    Variables:
        arr (NDArray[Any]): Working NumPy array converted to float64 for precision during processing.
        divisor (int): Greatest common divisor between sample_rate and 16000 used for polyphase resampling ratios.
        info (Any): Integer type bounds metadata (np.iinfo) used to scale signed/unsigned integer PCM into [-1, 1].

    Returns:
        NDArray[np.float32]: Contiguous 1-D mono float32 waveform normalized to [-1.0, 1.0] at 16 kHz.

    Raises:
        InvalidAudioError: If sample_rate is invalid, or if the array is empty, non-numeric, has unsupported
            dimensions/channel counts, contains NaN/Inf values, or exceeds [-1.0, 1.0] amplitude bounds.
    """
    arr: NDArray[Any]
    divisor: int
    info: Any

    # Verify sample_rate is a strictly positive integer (excluding booleans)
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise InvalidAudioError('sample_rate must be a positive integer')

    # Convert input into a NumPy array and validate dimensions and numeric dtype
    try:
        arr = np.asarray(samples)
    except Exception as exc:
        raise InvalidAudioError('invalid array') from exc

    if arr.ndim not in (1, 2) or not arr.size or arr.dtype.kind not in 'fiu':
        raise InvalidAudioError('expected a nonempty numeric mono or samples-by-channels array')
    if arr.ndim == 2 and (arr.shape[1] < 1 or arr.shape[1] > 8):
        raise InvalidAudioError('2-D audio must have samples on axis 0 and 1–8 channels on axis 1')

    # Scale integer PCM formats (signed or unsigned) to [-1.0, 1.0] float64, or cast float inputs
    if arr.dtype.kind in 'iu':
        info = np.iinfo(arr.dtype)
        if arr.dtype.kind == 'u':
            arr = (arr.astype(np.float64) - (info.max + 1) / 2) / ((info.max + 1) / 2)
        else:
            arr = arr.astype(np.float64) / max(abs(info.min), info.max)
    else:
        arr = arr.astype(np.float64)

    # Check for NaN or infinite sample values
    if not np.all(np.isfinite(arr)):
        raise InvalidAudioError('audio contains non-finite samples')

    # Downmix multi-channel audio to mono by averaging across channels (axis 1)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)

    # Resample to 16 kHz using polyphase filtering when the input sample rate differs
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        arr = resample_poly(arr, 16000 // divisor, sample_rate // divisor)

    # Verify floating-point amplitude lies within [-1.0, 1.0] (with small tolerance)
    if np.any(np.abs(arr) > 1.0 + 1e-6):
        raise InvalidAudioError('floating-point audio must be normalized to [-1, 1]')

    return np.ascontiguousarray(arr, dtype=np.float32)


# Inspect a single audio file or NumPy array without running GPU inference
def inspect_audio(
    audio: str | Path | NDArray[Any],
    *,
    sample_rate: int = 16000,
    audio_id: str | None = None,
    raise_on_error: bool = False,
) -> AudioInspection:
    """
    Decode an audio file or validate/resample an in-memory array on CPU, compute its
    duration, sample count, signal amplitude, and exact Mel content frame count
    (`len(wave) // 160`), and check whether it satisfies the `<= 1500` Mel-frame limit
    without loading or invoking the GPU Whisper model.

    Args:
        audio (str | Path | NDArray[Any]): Path to an audio file or a numeric NumPy audio array.
        sample_rate (int): Sampling rate in Hz when `audio` is a NumPy array (ignored for file paths).
            Defaults to 16000.
        audio_id (str | None): Optional identifier override (defaults to file stem or `'array'`).
        raise_on_error (bool): When True, immediately raises `InvalidAudioError` or
            `UnsupportedAudioLengthError` if the audio is invalid or exceeds 1500 Mel frames.
            When False (default), captures the error inside the returned `AudioInspection`.

    Variables:
        resolved_id (str): Effective identifier for the inspected audio item.
        source_path (str | None): Resolved filesystem path string if `audio` is a path, else None.
        wave (NDArray[np.float32]): Decoded and normalized 16 kHz mono float32 waveform.
        num_samples (int): Total number of 16 kHz mono samples in `wave`.
        duration (float): Audio duration in seconds (`num_samples / 16000.0`).
        actual_frames (int): Number of Mel spectrogram content frames (`num_samples // 160`).
        peak (float): Maximum absolute sample value in `wave`.
        rms (float): Root-mean-square amplitude of `wave`.
        msg (str): Validation failure message when frame count is `<= 0` or `> 1500`.

    Returns:
        AudioInspection: Structured inspection result detailing eligibility, frame counts, and signal stats.

    Raises:
        InvalidAudioError: If `raise_on_error=True` and the audio cannot be decoded or is invalid.
        UnsupportedAudioLengthError: If `raise_on_error=True` and `actual_mel_frames` is `<= 0` or `> 1500`.
    """
    resolved_id: str
    source_path: str | None
    wave: NDArray[np.float32]
    num_samples: int
    duration: float
    actual_frames: int
    peak: float
    rms: float
    msg: str

    if isinstance(audio, (str, Path)):
        p = Path(audio)
        resolved_id = str(audio_id) if audio_id is not None else p.stem
        source_path = str(p.resolve()) if p.exists() else str(p)
    else:
        resolved_id = str(audio_id) if audio_id is not None else 'array'
        source_path = None

    try:
        if isinstance(audio, (str, Path)):
            wave = decode_file(audio)
        elif isinstance(audio, np.ndarray):
            wave = prepare_array(audio, sample_rate)
        else:
            raise InvalidAudioError('audio must be a path or NumPy array')
    except InvalidAudioError as exc:
        if raise_on_error:
            raise
        return AudioInspection(
            audio_id=resolved_id,
            source=source_path,
            valid=False,
            status='invalid_audio',
            duration=0.0,
            num_samples=0,
            actual_mel_frames=0,
            max_allowed_frames=MAX_CONTENT_FRAMES,
            peak_amplitude=None,
            rms_amplitude=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    num_samples = int(len(wave))
    duration = num_samples / 16000.0
    actual_frames = num_samples // 160
    peak = float(np.max(np.abs(wave))) if num_samples > 0 else 0.0
    rms = float(np.sqrt(np.mean(np.square(wave, dtype=np.float64)))) if num_samples > 0 else 0.0

    if actual_frames <= 0:
        msg = 'audio has no Mel content frames'
        if raise_on_error:
            raise UnsupportedAudioLengthError(msg)
        return AudioInspection(
            audio_id=resolved_id,
            source=source_path,
            valid=False,
            status='empty_mel_frames',
            duration=duration,
            num_samples=num_samples,
            actual_mel_frames=actual_frames,
            max_allowed_frames=MAX_CONTENT_FRAMES,
            peak_amplitude=peak,
            rms_amplitude=rms,
            error_type='UnsupportedAudioLengthError',
            error_message=msg,
        )

    if actual_frames > MAX_CONTENT_FRAMES:
        msg = (
            f'audio length ({actual_frames} Mel frames) exceeds the maximum supported limit of '
            f'{MAX_CONTENT_FRAMES} Mel frames (~15 seconds); long-form chunking for inputs '
            f'above {MAX_CONTENT_FRAMES} Mel frames is planned for future work'
        )
        if raise_on_error:
            raise UnsupportedAudioLengthError(msg)
        return AudioInspection(
            audio_id=resolved_id,
            source=source_path,
            valid=False,
            status='exceeds_1500_frames',
            duration=duration,
            num_samples=num_samples,
            actual_mel_frames=actual_frames,
            max_allowed_frames=MAX_CONTENT_FRAMES,
            peak_amplitude=peak,
            rms_amplitude=rms,
            error_type='UnsupportedAudioLengthError',
            error_message=msg,
        )

    return AudioInspection(
        audio_id=resolved_id,
        source=source_path,
        valid=True,
        status='eligible',
        duration=duration,
        num_samples=num_samples,
        actual_mel_frames=actual_frames,
        max_allowed_frames=MAX_CONTENT_FRAMES,
        peak_amplitude=peak,
        rms_amplitude=rms,
        error_type=None,
        error_message=None,
    )


# Inspect a directory of audio files or an iterable of audio inputs without running GPU inference
def inspect_dataset(
    dataset: str | Path | Iterable[str | Path | NDArray[Any]],
    *,
    pattern: str | Sequence[str] | None = None,
    recursive: bool = False,
    sample_rate: int = 16000,
) -> DatasetInspection:
    """
    Scan and validate an entire audio dataset (either a directory path on disk or an iterable
    of file paths / NumPy arrays) on CPU without loading or running the GPU Whisper model.
    Identifies which files are eligible (`<= 1500` Mel frames), which exceed the 1500-frame
    short-form limit, and which are unreadable or corrupt.

    Args:
        dataset (str | Path | Iterable[str | Path | NDArray[Any]]): Either a directory path
            containing audio files or an iterable of audio file paths / NumPy arrays.
        pattern (str | Sequence[str] | None): Glob pattern(s) used when `dataset` is a directory
            (defaults to `('*.wav', '*.flac', '*.mp3', '*.ogg', '*.m4a')`).
        recursive (bool): When True and `dataset` is a directory, searches subdirectories recursively.
            Defaults to False.
        sample_rate (int): Sampling rate in Hz for any in-memory NumPy arrays in `dataset`.
            Defaults to 16000.

    Variables:
        targets (list[str | Path | NDArray[Any]]): Ordered list of audio items to inspect.
        patterns (tuple[str, ...]): Normalized glob patterns used for directory discovery.
        discovered (dict[str, Path]): Deduplicated mapping of resolved paths discovered in a directory.
        items (list[AudioInspection]): Per-item `AudioInspection` results.
        decoded_frames (list[int]): Mel frame counts of all successfully decoded items (`num_samples > 0`).
        total_count (int): Total number of inspected items.
        eligible_count (int): Number of items with `status == 'eligible'`.
        exceeds_count (int): Number of items with `status == 'exceeds_1500_frames'`.
        invalid_count (int): Number of items with `'invalid_audio'` or `'empty_mel_frames'`.
        total_dur (float): Total duration in seconds across all decoded items.
        eligible_dur (float): Total duration in seconds across all eligible items.

    Returns:
        DatasetInspection: Aggregated dataset health report, statistics, and per-item inspection list.

    Raises:
        ConfigurationError: If `dataset` is a path that does not exist or is neither a directory nor a file.
    """
    targets: list[str | Path | NDArray[Any]]
    patterns: tuple[str, ...]
    discovered: dict[str, Path]
    items: list[AudioInspection]
    decoded_frames: list[int]
    total_count: int
    eligible_count: int
    exceeds_count: int
    invalid_count: int
    total_dur: float
    eligible_dur: float

    if isinstance(dataset, (str, Path)):
        root = Path(dataset)
        if not root.exists():
            raise ConfigurationError(f'dataset path not found: {root}')
        if root.is_file():
            targets = [root]
        elif root.is_dir():
            if pattern is None:
                patterns = DEFAULT_AUDIO_PATTERNS
            elif isinstance(pattern, str):
                patterns = (pattern,)
            else:
                patterns = tuple(pattern)
            discovered = {}
            for pat in patterns:
                matches = root.rglob(pat) if recursive else root.glob(pat)
                for match in sorted(matches):
                    if match.is_file():
                        discovered[str(match.resolve())] = match
            targets = list(discovered.values())
        else:
            raise ConfigurationError(f'unsupported dataset path: {root}')
    elif isinstance(dataset, Iterable) and not isinstance(dataset, (bytes, bytearray)):
        targets = list(dataset)
    else:
        raise ConfigurationError('dataset must be a directory path, file path, or iterable of audio inputs')

    items = []
    for idx, entry in enumerate(targets):
        aid = None if isinstance(entry, (str, Path)) else f'array_{idx}'
        items.append(inspect_audio(entry, sample_rate=sample_rate, audio_id=aid, raise_on_error=False))

    total_count = len(items)
    eligible_count = sum(1 for item in items if item.status == 'eligible')
    exceeds_count = sum(1 for item in items if item.status == 'exceeds_1500_frames')
    invalid_count = total_count - eligible_count - exceeds_count

    total_dur = sum(item.duration for item in items)
    eligible_dur = sum(item.duration for item in items if item.valid)
    decoded_frames = [item.actual_mel_frames for item in items if item.num_samples > 0]

    return DatasetInspection(
        total_count=total_count,
        eligible_count=eligible_count,
        exceeds_1500_count=exceeds_count,
        invalid_count=invalid_count,
        eligible_ratio=(eligible_count / total_count) if total_count > 0 else 0.0,
        total_duration_seconds=total_dur,
        eligible_duration_seconds=eligible_dur,
        min_mel_frames=min(decoded_frames) if decoded_frames else None,
        mean_mel_frames=float(statistics.mean(decoded_frames)) if decoded_frames else None,
        max_mel_frames=max(decoded_frames) if decoded_frames else None,
        items=items,
    )


