"""faster-whisper 1.2.1 integration; exact encoder shapes are checked at runtime."""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import threading
import time
from pathlib import Path
from collections.abc import Iterator
from typing import Any
import numpy as np
from numpy.typing import NDArray
from ..config import WhisperConfig
from ..exceptions import ModelLoadError, InferenceError, ModelCompatibilityError

# faster-whisper uses a module-global pad_or_trim and CTranslate2's RNG is global.
# All instances of this library serialize those operations; unrelated code must not
# call faster-whisper concurrently while one of these windows is active.
GLOBAL_LOCK = threading.RLock()

# Expected SHA-256 digests for verified auxiliary model configuration files
MODEL_FILES: dict[str, str] = {
    'config.json': '27ec8ecb8ba05f3ac81522a7a035bd5ab19a5b567640bf68ccec90ded4fc1726',
    'preprocessor_config.json': '639427792a627557c19c95880aad248e591d8b57ef61e6dfa4e8249e353a0416',
    'vocabulary.json': '697ee5a65726e46b4d294d1a243d98e1878f5d81caba17891b1c5ebab7a912a9',
    'tokenizer.json': '16f415e176fbabcf63148d75d11f0a430e391c34603a57b588550a83f7eb798e',
}

# Expected byte size and sampled SHA-256 fingerprint for the verified model.bin file
MODEL_BIN_SIZE: int = 820316291
MODEL_BIN_FINGERPRINT: str = '4997255035685d2e12be5a42cffa34b41c1b98cceb85ee483b53f259e6b5309e'


# Compute the full SHA-256 hexadecimal digest of a file in 1 MiB chunks
def file_hash(path: Path) -> str:
    """
    Read a file from disk in 1 MiB binary blocks and compute its SHA-256 hex digest.

    Args:
        path (Path): Filesystem path to the file to hash.

    Variables:
        block (bytes): Current 1 MiB binary chunk read from the file stream.
        f (Any): Opened binary file handle.
        h (Any): Incremental hashlib SHA-256 hash object.

    Returns:
        str: Hexadecimal SHA-256 digest string of the file contents.
    """
    block: bytes
    f: Any
    h: Any
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# Verify local CTranslate2 model files and compute a fast sampled fingerprint of model.bin
def model_identity(path: str | Path) -> tuple[bool, str]:
    """
    Check whether a local CTranslate2 model directory matches the verified Experiment 05/06
    model hashes and compute a fast SHA-256 fingerprint of `model.bin` (sampling beginning,
    middle, and end 1 MiB blocks for large weights files).

    Args:
        path (str | Path): Path to the local CTranslate2 model directory.

    Variables:
        f (Any): Binary file stream opened on `model.bin`.
        file (Path): Resolved path to `model.bin` inside the model directory.
        fingerprint (str): Hexadecimal SHA-256 digest representing `model.bin` size and sampled chunks.
        h (Any): SHA-256 hasher accumulating the `model.bin` size header and sampled byte slices.
        hashes (dict[str, str | None]): Actual SHA-256 digests of the JSON metadata files in `MODEL_FILES`.
        offset (int): Byte offset used when seeking to start, midpoint, and tail of `model.bin`.
        size (int): Total byte size of `model.bin` on disk.
        valid (bool): True if all JSON hashes, `model.bin` byte size, and fingerprint match verified constants.

    Returns:
        tuple[bool, str]: A tuple of (is_verified_model, model_bin_fingerprint).

    Raises:
        ModelLoadError: If `model.bin` or the model directory cannot be accessed on disk.
    """
    f: Any
    file: Path
    fingerprint: str
    h: Any
    hashes: dict[str, str | None]
    offset: int
    size: int
    valid: bool
    path = Path(path)
    try:
        # Hash all expected JSON configuration and vocabulary files
        hashes = {
            name: file_hash(path / name) if (path / name).is_file() else None
            for name in MODEL_FILES
        }

        # Inspect model.bin size and hash up to three 1 MiB slices (start, middle, end)
        file = path / 'model.bin'
        size = file.stat().st_size
        h = hashlib.sha256()
        h.update(f'size:{size};'.encode())
        with file.open('rb') as f:
            if size <= 3 * 1024 * 1024:
                h.update(f.read())
            else:
                for offset in (0, size // 2 - 512 * 1024, max(0, size - 1024 * 1024)):
                    f.seek(offset)
                    h.update(f.read(1024 * 1024))

        fingerprint = h.hexdigest()
        valid = (
            hashes == MODEL_FILES
            and size == MODEL_BIN_SIZE
            and fingerprint == MODEL_BIN_FINGERPRINT
        )
        return valid, fingerprint
    except OSError as exc:
        raise ModelLoadError(f'incomplete local CTranslate2 model: {exc}') from exc


# Derive a deterministic 32-bit integer RNG seed from base seed, audio ID, and frame window
def seed_for(seed: int, audio_id: str, frames: int) -> int:
    """
    Compute a reproducible 32-bit unsigned integer seed for CTranslate2 by hashing the
    combination of user seed, audio identifier, and target encoder frame count.

    Args:
        seed (int): Base integer seed from InferenceConfig.
        audio_id (str): Identifier string for the audio being transcribed.
        frames (int): Target encoder window size in Mel frames (e.g., 1500 or 3000).

    Returns:
        int: Deterministic 32-bit unsigned integer derived from the first 4 bytes of SHA-256.
    """
    return int.from_bytes(
        hashlib.sha256(f'{seed}|{audio_id}|{frames}'.encode()).digest()[:4],
        'big',
    )


# Convert a perf_counter_ns start timestamp into elapsed milliseconds
def elapsed(start: int) -> float:
    """
    Compute elapsed wall-clock time in milliseconds since a `time.perf_counter_ns()` timestamp.

    Args:
        start (int): Starting timestamp in nanoseconds from `time.perf_counter_ns()`.

    Returns:
        float: Elapsed time in milliseconds.
    """
    return (time.perf_counter_ns() - start) / 1e6


# Context manager that hooks faster-whisper to reuse cached Mel features and enforce exact encoder window size
@contextmanager
def controlled_window(model: Any, mel: NDArray[Any], target: int) -> Iterator[dict[str, Any]]:
    """
    Temporarily patch a faster-whisper `WhisperModel` instance and `faster_whisper.transcribe.pad_or_trim`
    so that:
      1. `model.feature_extractor` returns the pre-computed `mel` spectrogram without recomputation.
      2. `pad_or_trim` pads the spectrogram to `target` frames (1500 or 3000) and refuses truncation.
      3. `model.encode` and `model.generate_with_fallback` record exact input shapes and wall-clock timings.
    All patched attributes and module globals are restored on exit.

    Args:
        model (Any): Underlying `faster_whisper.WhisperModel` instance.
        mel (NDArray[Any]): Pre-extracted 2-D Mel spectrogram array of shape (128, N).
        target (int): Exact number of Mel frames (e.g., 1500 or 3000) to pad the encoder input to.

    Variables:
        cls (Any): Original class of `model.feature_extractor`, restored when exiting the context.
        extractor (Any): Feature extractor object attached to `model`.
        measured (dict[str, Any]): Mutable dictionary accumulating 'encoder_ms', 'decoder_ms',
            'encoder_shapes', and 'mel_calls' during the controlled execution window.
        original_decode (Any): Bound reference to `model.generate_with_fallback` before patching.
        original_encode (Any): Bound reference to `model.encode` before patching.
        original_pad (Any): Original `faster_whisper.transcribe.pad_or_trim` function before patching.
        sentinel (Any): Unique sentinel object used to check whether `encode` / `generate_with_fallback`
            were originally present in `model.__dict__`.
        stored_decode (Any): Original `model.__dict__` entry for `'generate_with_fallback'`, or `sentinel`.
        stored_encode (Any): Original `model.__dict__` entry for `'encode'`, or `sentinel`.

    Yields:
        dict[str, Any]: The `measured` telemetry dictionary populated during `model.transcribe()`.
    """
    cls: Any
    extractor: Any
    measured: dict[str, Any]
    original_decode: Any
    original_encode: Any
    original_pad: Any
    sentinel: Any
    stored_decode: Any
    stored_encode: Any
    import faster_whisper.transcribe as fw

    # Capture original extractor class, module pad_or_trim function, and model methods
    extractor = model.feature_extractor
    cls = extractor.__class__
    original_pad = fw.pad_or_trim
    original_encode = model.encode
    original_decode = model.generate_with_fallback

    # Track whether encode and generate_with_fallback existed on the instance dict directly
    sentinel = object()
    stored_encode = model.__dict__.get('encode', sentinel)
    stored_decode = model.__dict__.get('generate_with_fallback', sentinel)

    # Initialize telemetry accumulator for the current inference pass
    measured = {
        'encoder_ms': 0.0,
        'decoder_ms': 0.0,
        'encoder_shapes': [],
        'mel_calls': 0,
    }

    # Subclass of the feature extractor that returns the cached Mel array instead of recomputing
    class CachedExtractor(cls):
        """
        Drop-in replacement class for `model.feature_extractor` that increments `mel_calls`
        and returns the pre-computed Mel spectrogram array.
        """

        # Return the pre-computed Mel spectrogram on call
        def __call__(self, *args: Any, **kwargs: Any) -> NDArray[Any]:
            """
            Record the feature extraction invocation and return the cached `mel` array.
            """
            measured['mel_calls'] += 1
            return mel

    # Custom pad_or_trim replacement that pads up to `target` frames and forbids truncation
    def pad(array: NDArray[Any], length: int = 3000, *, axis: int = -1) -> NDArray[Any]:
        """
        Pad `array` with zeros along `axis` to reach `target` frames, raising InferenceError
        if the input array already exceeds `target` frames.

        Args:
            array (NDArray[Any]): Input Mel spectrogram slice to pad.
            length (int): Default length parameter from faster-whisper (ignored in favor of `target`).
            axis (int): Axis along which frames are counted and padded (defaults to -1).

        Variables:
            size (int): Current frame count of `array` along `axis`.
            widths (list[tuple[int, int]]): Per-dimension (before, after) padding widths for `np.pad`.

        Returns:
            NDArray[Any]: Zero-padded array of length `target` along `axis`.

        Raises:
            InferenceError: If `size > target`, preventing silent audio truncation.
        """
        size: int
        widths: list[tuple[int, int]]
        size = int(array.shape[axis])
        if size > target:
            raise InferenceError(f'input would be truncated: {size} > {target} frames')
        widths = [(0, 0)] * array.ndim
        widths[axis] = (0, target - size)
        return np.pad(array, widths, mode='constant') if size < target else array

    # Wrapper around model.encode that records input tensor shapes and encoder execution time
    def encode(features: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Record the shape of `features` passed to the Whisper encoder and measure elapsed encoder time.

        Args:
            features (Any): Padded Mel spectrogram tensor passed into `model.encode`.

        Variables:
            start (int): Nanosecond timestamp recorded immediately before calling `original_encode`.

        Returns:
            Any: Encoder output returned by `original_encode`.
        """
        start: int
        measured['encoder_shapes'].append(tuple(int(v) for v in features.shape))
        start = time.perf_counter_ns()
        try:
            return original_encode(features, *args, **kwargs)
        finally:
            measured['encoder_ms'] += elapsed(start)

    # Wrapper around model.generate_with_fallback that measures decoder execution time
    def decoder(*args: Any, **kwargs: Any) -> Any:
        """
        Invoke `original_decode` (`model.generate_with_fallback`) and accumulate elapsed decoder time.

        Variables:
            start (int): Nanosecond timestamp recorded immediately before calling `original_decode`.

        Returns:
            Any: Decoder output returned by `original_decode`.
        """
        start: int
        start = time.perf_counter_ns()
        try:
            return original_decode(*args, **kwargs)
        finally:
            measured['decoder_ms'] += elapsed(start)

    # Apply runtime patches for the duration of the context block and restore them unconditionally
    try:
        extractor.__class__ = CachedExtractor
        fw.pad_or_trim = pad
        model.encode = encode
        model.generate_with_fallback = decoder
        yield measured
    finally:
        extractor.__class__ = cls
        fw.pad_or_trim = original_pad
        if stored_encode is sentinel:
            model.__dict__.pop('encode', None)
        else:
            model.encode = stored_encode
        if stored_decode is sentinel:
            model.__dict__.pop('generate_with_fallback', None)
        else:
            model.generate_with_fallback = stored_decode


# Low-level CTranslate2 / faster-whisper backend managing model lifecycle, Mel extraction, and windowed decoding
class CTranslateBackend:
    """
    Wraps `faster_whisper.WhisperModel` (strictly v1.2.1) to provide isolated Mel spectrogram
    extraction and shape-verified single-window encoder/decoder execution.

    Attributes:
        config (WhisperConfig): Model and decoding configuration parameters.
        model (Any): Loaded `faster_whisper.WhisperModel` instance, or None after `close()`.
        mel_bins (int): Number of Mel frequency bins verified on the model's feature extractor (128).
    """

    # Initialize the faster-whisper model from local files and validate version and Mel filterbank shape
    def __init__(self, config: WhisperConfig) -> None:
        """
        Verify that `faster-whisper == 1.2.1` is installed, load the local CTranslate2 WhisperModel,
        and confirm that the model uses a 128-bin Mel filterbank.

        Args:
            config (WhisperConfig): Validated Whisper model and device configuration.

        Raises:
            ModelCompatibilityError: If `faster-whisper` is not version 1.2.1 or Mel bins != 128.
            ModelLoadError: If `WhisperModel` fails to load from `config.model_path`.
        """
        from importlib.metadata import version
        from faster_whisper import WhisperModel

        # Enforce exact faster-whisper version required by controlled_window hooks
        if version('faster-whisper') != '1.2.1':
            raise ModelCompatibilityError('faster-whisper 1.2.1 is required by the shape-control integration')

        self.config: WhisperConfig = config

        # Load the CTranslate2 model strictly from local disk files
        try:
            self.model = WhisperModel(
                config.model_path,
                device=config.device,
                device_index=config.device_index,
                compute_type=config.compute_type,
                cpu_threads=config.cpu_threads,
                num_workers=config.num_workers,
                local_files_only=True,
            )
        except Exception as exc:
            raise ModelLoadError(f'could not load local model: {exc}') from exc

        # Inspect and verify that the model expects 128 Mel frequency bins (Whisper large-v3 / turbo)
        try:
            self.mel_bins: int = int(self.model.feature_extractor.mel_filters.shape[0])
        except (AttributeError, TypeError, ValueError):
            self.close()
            raise ModelCompatibilityError('cannot identify model Mel bin count')
        if self.mel_bins != 128:
            self.close()
            raise ModelCompatibilityError(f'expected 128 Mel bins, got {self.mel_bins}')

    # Extract a 128-bin log-Mel spectrogram from a 16 kHz float32 audio waveform
    def extract(self, wave: NDArray[np.float32]) -> NDArray[Any]:
        """
        Compute the log-Mel spectrogram from a normalized 16 kHz waveform using the model's
        feature extractor and validate its 2-D shape `(128, frames)`.

        Args:
            wave (NDArray[np.float32]): 1-D float32 audio waveform sampled at 16 kHz.

        Variables:
            mel (NDArray[Any]): Extracted 2-D log-Mel spectrogram array returned by `feature_extractor`.

        Returns:
            NDArray[Any]: Validated 2-D Mel spectrogram array with 128 frequency bins and >= 2 frames.

        Raises:
            InferenceError: If the backend is closed, extraction raises an error, or output shape is invalid.
        """
        mel: NDArray[Any]
        if self.model is None:
            raise InferenceError('model is closed')
        try:
            mel = self.model.feature_extractor(wave)
            if getattr(mel, 'ndim', None) != 2 or mel.shape[0] != 128 or mel.shape[-1] < 2:
                raise InferenceError(f'invalid Mel shape: {getattr(mel, "shape", None)}')
            return mel
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f'feature extraction failed: {exc}') from exc

    # Run a single controlled encoder/decoder pass at the specified Mel frame window size and RNG seed
    def run(
        self,
        wave: NDArray[np.float32],
        mel: NDArray[Any],
        frames: int,
        seed: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        Set the CTranslate2 random seed and execute `self.model.transcribe()` inside `controlled_window`
        so that `mel` is reused without recomputation and padded to `frames` (1500 or 3000).
        Verifies post-execution invariants (single Mel call, single encoder call of length `frames`,
        and single decoder window at `seek == 0`).

        Args:
            wave (NDArray[np.float32]): Original 16 kHz float32 waveform passed to `model.transcribe`.
            mel (NDArray[Any]): Pre-extracted Mel spectrogram reused via `controlled_window`.
            frames (int): Target encoder input length in Mel frames (1500 or 3000).
            seed (int): Deterministic seed passed to `ctranslate2.set_random_seed`.

        Variables:
            info (Any): Transcription info metadata returned by `faster_whisper` (unused directly).
            measured (dict[str, Any]): Telemetry dict with encoder/decoder timings and encoder shapes.
            segments (Any): Iterator and then materialized list of decoded segment objects.
            shapes (list[tuple[int, ...]]): List of tensor shapes observed by the patched `model.encode`.

        Returns:
            tuple[list[Any], dict[str, Any]]: A tuple of `(decoded_segments, measured_telemetry)`.

        Raises:
            InferenceError: If the model is closed, CTranslate2/faster-whisper fails, or runtime
                window/shape invariants are violated.
        """
        info: Any
        measured: dict[str, Any]
        segments: Any
        shapes: list[tuple[int, ...]]
        import ctranslate2

        if self.model is None:
            raise InferenceError('model is closed')
        try:
            # Seed CTranslate2 RNG deterministically and run transcription under the controlled window hook
            ctranslate2.set_random_seed(seed)
            with controlled_window(self.model, mel, frames) as measured:
                segments, info = self.model.transcribe(
                    wave,
                    language=self.config.language,
                    task=self.config.task,
                    beam_size=self.config.beam_size,
                    vad_filter=False,
                    word_timestamps=False,
                    without_timestamps=True,
                )
                segments = list(segments)

            # Verify that exactly one cached Mel lookup and one target-sized encoder call occurred
            shapes = measured['encoder_shapes']
            if measured['mel_calls'] != 1 or len(shapes) != 1 or shapes[0][-1] != frames:
                raise InferenceError(
                    f'expected one cached Mel and one {frames}-frame encoder call; got {measured}'
                )

            # Verify that decoding completed in a single window without advancing seek
            if len(segments) > 1 or any(s.seek != 0 for s in segments):
                raise InferenceError('expected a single decoder window and seek=0')

            return segments, measured
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f'inference failed: {exc}') from exc

    # Release the underlying WhisperModel reference
    def close(self) -> None:
        """
        Clear the reference to the underlying `faster_whisper.WhisperModel` instance
        so device memory can be reclaimed.
        """
        self.model = None

