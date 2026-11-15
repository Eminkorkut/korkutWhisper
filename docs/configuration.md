# Configuration & Gate Profiles

## 1. Composite Configuration (`KorkutConfig`)

`KorkutConfig` aggregates five frozen, validated dataclasses and supports full JSON round-tripping via `save_json(path)` and `KorkutConfig.load_json(path)`:

- **`WhisperConfig`**:
  - `model_path: str` (required path to local CTranslate2 model directory)
  - `device: str = 'cuda'` (`'cuda'`, `'cpu'`, or `'auto'`)
  - `compute_type: str = 'int8'`
  - `device_index: int = 0`
  - `cpu_threads: int = 0`
  - `num_workers: int = 1`
  - `beam_size: int = 5`
  - `language: str = 'tr'`
  - `task: str = 'transcribe'` (`'transcribe'` or `'translate'`)
- **`InferenceConfig`**:
  - `strategy: str = 'g2'` (`'g2'`, `'fixed_1500'`, or `'fixed_3000'`)
  - `seed: int = 42` (non-negative integer base seed for deterministic CTranslate2 RNG seeding)
- **`GateConfig`**:
  - `profile: str = 'decoder_based_b2'` (built-in profile identifier)
  - `allow_unverified: bool = False` (permits running `g2` on unverified models, runtime versions, or modified/custom gate profiles with a `RuntimeWarning`)
  - `profile_path: str | None = None` (optional path to an external custom JSON profile file)
- **`AudioConfig`**:
  - `sample_rate: int = 16000` (must be `16000` Hz)
- **`RuntimeConfig`**:
  - `max_concurrency: int = 1` (must be `1`; sequential execution)
  - `warmup: bool = False` (when `True`, runs a 1-second silent warm-up pass during `KorkutWhisper` initialization)

---

## 2. JSON Gate Profile (`decoder_based_b2.json`) & SHA-256 Verification

Calibration parameters for the G2 decoder-risk gate are stored in [`src/korkutwhisper/profiles/decoder_based_b2.json`](../src/korkutwhisper/profiles/decoder_based_b2.json):

```json
{
  "profile": "decoder_based_b2",
  "logprob_weight": -0.11143469276570268,
  "compression_weight": 0.03922713673281132,
  "logprob_mean": -0.06624447784135314,
  "compression_mean": 0.9736811585730087,
  "logprob_std": 0.06116229410922396,
  "compression_std": 0.11310338885374968,
  "logprob_median": -0.04859209232605421,
  "compression_median": 0.975609756097561,
  "threshold": 0.07826304331214909,
  "decision": "risk_score >= threshold",
  "score_is_probability": false
}
```

### Cryptographic Integrity Check
- `korkutwhisper.profiles.VERIFIED_PROFILE_HASHES` stores the expected SHA-256 digest (`70e68351003baabf12a6371c32c155aaa0d44f4eb535e141e189aac0c9d23e37`) of `decoder_based_b2.json` (computed over LF-normalized UTF-8 bytes so it is platform-independent across Windows and Linux).
- When `load_gate_profile()` loads the JSON file, it verifies all numeric fields and compares the file's SHA-256 digest against `VERIFIED_PROFILE_HASHES`.
- Full profile validation (`model._validated == True`) requires:
  1. Verified local model file hashes and sampled `model.bin` fingerprint,
  2. Verified `decoder_based_b2.json` SHA-256 hash (`gate_profile.verified == True`),
  3. `device == 'cuda'`, `compute_type == 'int8'`, `language == 'tr'`, `task == 'transcribe'`, `beam_size == 5`,
  4. `faster-whisper == 1.2.1` and `ctranslate2 == 4.8.3`.
- Any deviation requires `allow_unverified=True` when `strategy == 'g2'`.
