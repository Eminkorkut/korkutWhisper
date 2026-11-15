# API Reference

## 1. Primary Client: `KorkutWhisper`

```python
KorkutWhisper(
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
)
```

### Instance Methods
- **`warmup(duration_seconds: float = 1.0, *, include_retry_window: bool = False) -> float`**  
  Executes a synthetic silent pass (between `0.1` and `15.0` seconds) to pre-initialize CUDA/CTranslate2 kernels before timing real audio. When `include_retry_window=True`, warms up both `1500`-frame and `3000`-frame encoder windows. Returns elapsed warm-up time in milliseconds.
- **`transcribe(audio: str | Path | NDArray[Any], *, audio_id: str | None = None) -> TranscriptionResult`**  
  Decodes an audio file path or validates a 16 kHz NumPy array (`<= 1500` Mel content frames) and executes the configured transcription pipeline.
- **`transcribe_array(samples: Any, sample_rate: int = 16000, *, audio_id: str = 'array') -> TranscriptionResult`**  
  Validates, downmixes (`1–8` channels to mono), normalizes, and resamples a numeric array from `sample_rate` to 16 kHz before transcribing.
- **`transcribe_many(audios: Iterable[str | Path | NDArray[Any]], *, max_concurrency: int = 1) -> list[BatchItem]`**  
  Sequentially transcribes an iterable of audio inputs, capturing per-item `KorkutWhisperError` exceptions into `BatchItem` objects without aborting the batch.
- **`inspect_audio(audio: str | Path | NDArray[Any], *, sample_rate: int = 16000, audio_id: str | None = None, raise_on_error: bool = False) -> AudioInspection`**  
  Validates a single audio input on CPU and computes its duration, signal amplitude, and Mel frame count (`<= 1500` check) without running GPU inference.
- **`inspect_dataset(dataset: str | Path | Iterable[...], *, pattern: str | Sequence[str] | None = None, recursive: bool = False, sample_rate: int = 16000) -> DatasetInspection`**  
  Scans a directory or iterable of audio inputs on CPU and returns aggregate eligibility counts and Mel-frame statistics.
- **`get_model_info() -> ModelInfo`**  
  Returns metadata and verification status for the loaded model.
- **`get_config() -> dict[str, Any]`**  
  Returns the serialized `KorkutConfig` dictionary.
- **`get_gate_profile() -> dict[str, Any]`**  
  Returns the active `G2Gate` profile parameters, SHA-256 digest, and verification flag.
- **`close() -> None`** / **`__enter__()`** / **`__exit__()`**  
  Releases the underlying backend reference under `GLOBAL_LOCK`.

---

## 2. Standalone Functions

- **`inspect_audio(audio, *, sample_rate=16000, audio_id=None, raise_on_error=False) -> AudioInspection`**  
  CPU-only single-audio inspection without instantiating `KorkutWhisper`.
- **`inspect_dataset(dataset, *, pattern=None, recursive=False, sample_rate=16000) -> DatasetInspection`**  
  CPU-only dataset directory or iterable inspection without instantiating `KorkutWhisper`.
- **`load_gate_profile(profile='decoder_based_b2', profile_path=None) -> GateProfile`**  
  Loads a G2 JSON profile and verifies its SHA-256 digest against `VERIFIED_PROFILE_HASHES`.
- **`profile_hash(path) -> str`**  
  Computes the LF-normalized SHA-256 hex digest of a profile JSON file.
- **`format_batch_csv(items) -> str`** / **`save_batch_csv(items, path) -> Path`**  
  Formats or saves a sequence of `BatchItem` or `TranscriptionResult` objects as a 26-column CSV table (`CSV_COLUMNS`).
- **`save_batch_jsonl(items, path) -> Path`**  
  Saves a sequence of `BatchItem` or `TranscriptionResult` objects to a UTF-8 JSON Lines (`.jsonl`) file.
- **`cli_main(argv=None) -> int`**  
  Entry point for the `korkutwhisper` command-line interface.

---

## 3. Result Dataclasses (`korkutwhisper.types`)

All result dataclasses inherit from `JsonResult`, which provides:
- `to_dict() -> dict[str, Any]`
- `to_json(*, indent: int | None = 2) -> str`
- `save_json(path: str | Path, *, indent: int | None = 2) -> Path`

### `TranscriptionResult`
Fields: `text`, `language`, `duration`, `strategy`, `retry_performed`, `final_stage`, `segments`, `gate`, `timing`, `metadata`.  
Additional formatting and export methods:
- `to_text() -> str`: Plain transcribed text.
- `to_report() -> str`: Multi-line human-readable diagnostic summary card.
- `to_csv_row(*, index: int = 0) -> dict[str, Any]`: Flattened 26-column dictionary (`CSV_COLUMNS`).
- `to_csv(*, include_header: bool = True) -> str`: RFC-4180 CSV string.
- `to_srt() -> str`: SubRip (`.srt`) subtitle string.
- `to_vtt() -> str`: WebVTT (`.vtt`) subtitle string.
- `format(fmt: str = 'text') -> str`: Renders `'text'`, `'txt'`, `'report'`, `'json'`, `'csv'`, `'srt'`, or `'vtt'`.
- `save(path: str | Path, fmt: str | None = None) -> Path`: Saves formatted output to disk (inferring format from file extension if `fmt` is omitted).

### `AudioInspection` & `DatasetInspection`
- `AudioInspection`: `audio_id`, `source`, `valid`, `status`, `duration`, `num_samples`, `actual_mel_frames`, `max_allowed_frames`, `peak_amplitude`, `rms_amplitude`, `error_type`, `error_message`, plus `.to_report()` and `.to_csv_row()`.
- `DatasetInspection`: `total_count`, `eligible_count`, `exceeds_1500_count`, `invalid_count`, `eligible_ratio`, `total_duration_seconds`, `eligible_duration_seconds`, `min_mel_frames`, `mean_mel_frames`, `max_mel_frames`, `items`, plus `.eligible_items()`, `.rejected_items()`, `.eligible_paths()`, `.to_report()`, `.to_csv()`, and `.save_csv(path)`.

---

## 4. Public Exception Hierarchy (`korkutwhisper.exceptions`)

All exceptions inherit from `KorkutWhisperError`:
- `ConfigurationError`: Invalid configuration parameters, unknown format names, or malformed profile JSON.
- `ModelLoadError`: Missing or unreadable local CTranslate2 model files.
- `ModelCompatibilityError`: Unverified model/runtime/profile under `g2` without `allow_unverified=True`, incompatible `faster-whisper` version, or non-128 Mel bin count.
- `InvalidAudioError`: Missing audio file, corrupt audio frames, or invalid/non-finite NumPy array.
- `UnsupportedAudioLengthError`: Audio has `<= 0` Mel content frames or exceeds the maximum `1500` Mel-frame (~15s) short-form limit.
- `InferenceError`: Transcription attempted on a closed instance, or runtime Mel/encoder/decoder failure.
