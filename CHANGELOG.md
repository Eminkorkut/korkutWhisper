# Changelog

## 0.1.0
- Initial decoder-guided selective re-inference (`g2`) API and `CTranslate2` / `faster-whisper 1.2.1` shape-verified backend integration.
- Externalized G2 calibration profile into `profiles/decoder_based_b2.json` with platform-independent SHA-256 hash verification (`GateProfile`, `load_gate_profile`, `VERIFIED_PROFILE_HASHES`).
- Enforced strict short-form input bound (`1 <= actual_mel_frames <= 1500`, ~15 seconds) across all strategies with descriptive `UnsupportedAudioLengthError` diagnostics and documented future long-form chunking architecture.
- Added model warm-up support (`model.warmup()` and `KorkutWhisper(..., warmup=True)` / `RuntimeConfig(warmup=True)`) to eliminate first-run CUDA initialization latency spikes.
- Added CPU-only pre-inference audio and dataset inspection utilities (`inspect_audio`, `inspect_dataset`, `AudioInspection`, `DatasetInspection`).
- Added rich output formatters and file exporters (`to_text`, `to_report`, `to_json`, `to_csv`, `to_srt`, `to_vtt`, `format`, `save`, `format_batch_csv`, `save_batch_csv`, `save_batch_jsonl`).
- Added command-line interface (`korkutwhisper` / `python -m korkutwhisper`) supporting single/batch transcription, `--inspect`, `--gate-info`, and `--model-info`.
- Formatted all library, example, and test modules for readability with structured English heading comments and comprehensive docstrings.
