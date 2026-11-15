# Limitations & Scope

- **Short-Form Audio Scope (`<= 1500` Mel Frames):**  
  The current release strictly accepts audio inputs with `1 <= actual_mel_frames <= 1500` (~15 seconds at 16 kHz). Any audio exceeding `1500` Mel content frames is rejected with `UnsupportedAudioLengthError` across all strategies (`g2`, `fixed_1500`, `fixed_3000`). Nothing is silently truncated. Long-form audio chunking (`> 1500` frames) is planned for future work. Use `inspect_audio()` or `inspect_dataset()` to pre-screen audio files on CPU before running GPU inference.
- **Process-Global Hook Serialization:**  
  Only `faster-whisper==1.2.1` is supported. Because `controlled_window` temporarily patches `faster_whisper.transcribe.pad_or_trim` and CTranslate2's RNG seed is process-global, all calls within this library are serialized via `GLOBAL_LOCK`. Uncoordinated external `faster-whisper` calls in other threads within the same Python process can interfere with padding patches.
- **128 Mel Filterbank Requirement:**  
  The backend requires 128 Mel frequency bins (Whisper `large-v3` / `turbo` family); 80-bin checkpoints are rejected.
- **Segment Timestamps in Single-Window Mode:**  
  Backend decoding uses `without_timestamps=True` and a single encoder/decoder window (`seek=0`). Segment `start` and `end` fields are `None`, and `.srt` / `.vtt` exports span `00:00:00 --> duration` for the clip.
- **Calibration Specificity (`decoder_based_b2`):**  
  The G2 risk gate weights and threshold in `profiles/decoder_based_b2.json` (verified by SHA-256 hash) were calibrated specifically for the Experiment 05/06 Turkish INT8 model on CUDA with beam size 5. Applying `g2` to other models, languages, compute types, or custom JSON profiles requires `allow_unverified=True` and is experimental.
- **Sampled Model Binary Fingerprint:**  
  `model_identity()` hashes the full auxiliary JSON files (`config.json`, `preprocessor_config.json`, `vocabulary.json`, `tokenizer.json`) and three 1 MiB slices (start, middle, end) plus byte size of `model.bin` rather than hashing all ~820 MB on every load.
- **Host Wall-Clock Timing:**  
  `TimingResult` measures host wall-clock elapsed milliseconds without explicit `cudaDeviceSynchronize()` barriers. Use `warmup=True` or `model.warmup()` to eliminate first-run CUDA kernel initialization overhead before benchmarking.
- **Sequential Batch Execution:**  
  `transcribe_many()` runs sequentially (`max_concurrency=1`) and materializes a result list in memory.
