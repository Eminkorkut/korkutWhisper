"""Validated serializable configuration; no package installation is performed."""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import math
from typing import Any
from .exceptions import ConfigurationError


# Configuration for the underlying CTranslate2 / faster-whisper model and decoder
@dataclass(frozen=True)
class WhisperConfig:
    """
    Holds model loading and decoding parameters for the CTranslate2 Whisper backend.

    Attributes:
        model_path (str): Path to the local CTranslate2 model directory.
        device (str): Execution device ('cuda', 'cpu', or 'auto'). Defaults to 'cuda'.
        compute_type (str): Quantization/precision format (e.g., 'int8'). Defaults to 'int8'.
        device_index (int): GPU device index to use when running on CUDA. Defaults to 0.
        cpu_threads (int): Number of CPU threads (0 lets CTranslate2 choose automatically). Defaults to 0.
        num_workers (int): Number of CTranslate2 background worker threads. Defaults to 1.
        beam_size (int): Beam search width used during decoding. Defaults to 5.
        language (str): Target spoken language code. Defaults to 'tr'.
        task (str): Whisper operation mode ('transcribe' or 'translate'). Defaults to 'transcribe'.
    """
    model_path: str
    device: str = 'cuda'
    compute_type: str = 'int8'
    device_index: int = 0
    cpu_threads: int = 0
    num_workers: int = 1
    beam_size: int = 5
    language: str = 'tr'
    task: str = 'transcribe'

    # Validate Whisper model and decoding configuration fields after initialization
    def __post_init__(self) -> None:
        """
        Normalize model_path to a string and verify that all device, thread, beam,
        language, and task parameters fall within supported bounds.

        Raises:
            ConfigurationError: If any field has an unsupported or out-of-range value.
        """
        object.__setattr__(self, 'model_path', str(self.model_path))
        if not self.model_path or self.device not in ('cpu', 'cuda', 'auto') or not self.compute_type:
            raise ConfigurationError('model_path, device or compute_type is invalid')
        if self.device_index < 0 or self.cpu_threads < 0 or self.num_workers < 1 or self.beam_size < 1:
            raise ConfigurationError('device_index/cpu_threads must be nonnegative; workers/beam must be positive')
        if not self.language or self.task not in ('transcribe', 'translate'):
            raise ConfigurationError('language or task is invalid')


# Configuration for frame-window routing strategy and deterministic RNG seeding
@dataclass(frozen=True)
class InferenceConfig:
    """
    Controls the encoder window selection strategy and random seed used during inference.

    Attributes:
        strategy (str): Window routing strategy ('g2', 'fixed_1500', or 'fixed_3000'). Defaults to 'g2'.
        seed (int): Non-negative base integer seed used to derive per-audio CTranslate2 RNG seeds. Defaults to 42.
    """
    strategy: str = 'g2'
    seed: int = 42

    # Validate strategy name and random seed value
    def __post_init__(self) -> None:
        """
        Verify that the chosen inference strategy is recognized and the seed is a non-negative integer.

        Raises:
            ConfigurationError: If strategy is unknown or seed is not a non-negative integer.
        """
        if self.strategy not in ('g2', 'fixed_1500', 'fixed_3000') or not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigurationError('invalid inference strategy or seed')


# Configuration for the G2 decoder-risk gate profile and verification enforcement
@dataclass(frozen=True)
class GateConfig:
    """
    Specifies which G2 gate profile JSON to load and whether unverified setups or modified
    profile parameters are permitted.

    Attributes:
        profile (str): Identifier of the built-in G2 calibration profile. Defaults to 'decoder_based_b2'.
        allow_unverified (bool): When True, allows running G2 on unverified models/devices or custom/modified
            profile JSON files with a warning. Defaults to False.
        profile_path (str | None): Optional path to an external JSON profile file overriding the built-in directory.
            Defaults to None.
    """
    profile: str = 'decoder_based_b2'
    allow_unverified: bool = False
    profile_path: str | None = None

    # Validate that the requested gate profile exists and can be loaded
    def __post_init__(self) -> None:
        """
        Ensure the gate profile name or external `profile_path` resolves to a valid JSON profile.

        Raises:
            ConfigurationError: If `profile` is unknown when `profile_path` is None, or if the JSON is invalid.
        """
        if self.profile_path is not None:
            object.__setattr__(self, 'profile_path', str(self.profile_path))
        elif self.profile != 'decoder_based_b2':
            raise ConfigurationError('unknown gate profile')


# Configuration for audio preprocessing and target sampling rate
@dataclass(frozen=True)
class AudioConfig:
    """
    Defines expected audio signal characteristics for feature extraction.

    Attributes:
        sample_rate (int): Target audio sampling rate in Hz required by the Whisper frontend. Defaults to 16000.
    """
    sample_rate: int = 16000

    # Validate that the audio sample rate matches the 16 kHz Whisper requirement
    def __post_init__(self) -> None:
        """
        Check that sample_rate is set to 16000 Hz, as required by the Whisper feature extractor.

        Raises:
            ConfigurationError: If sample_rate is not 16000.
        """
        if self.sample_rate != 16000:
            raise ConfigurationError('backend requires 16 kHz')


# Configuration for runtime execution concurrency and optional model warm-up
@dataclass(frozen=True)
class RuntimeConfig:
    """
    Controls runtime concurrency limits and optional initialization warm-up for model execution.

    Attributes:
        max_concurrency (int): Maximum number of concurrent transcriptions allowed. Defaults to 1.
        warmup (bool): When True, automatically runs a silent warm-up pass upon initialization
            to pre-initialize CTranslate2/CUDA kernels before timing real audio. Defaults to False.
    """
    max_concurrency: int = 1
    warmup: bool = False

    # Validate runtime concurrency and warmup flag settings
    def __post_init__(self) -> None:
        """
        Enforce sequential execution (max_concurrency=1) because faster-whisper padding
        and CTranslate2 RNG states are module-global, and verify that warmup is a boolean.

        Raises:
            ConfigurationError: If max_concurrency is not equal to 1 or warmup is not a bool.
        """
        if self.max_concurrency != 1:
            raise ConfigurationError('this release supports sequential execution only (max_concurrency=1)')
        if not isinstance(self.warmup, bool):
            raise ConfigurationError('warmup must be a boolean')


# Top-level composite configuration aggregating all sub-configurations
@dataclass(frozen=True)
class KorkutConfig:
    """
    Master configuration object combining Whisper, inference, gate, audio, and runtime settings.

    Attributes:
        whisper (WhisperConfig): Model path, device, compute type, and decoding settings.
        inference (InferenceConfig): Routing strategy and deterministic seed configuration.
        gate (GateConfig): G2 risk gate profile and verification policy.
        audio (AudioConfig): Audio sampling rate configuration.
        runtime (RuntimeConfig): Execution concurrency configuration.
    """
    whisper: WhisperConfig
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # Convert the complete configuration tree into a nested dictionary
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this KorkutConfig and all nested config dataclasses into a Python dictionary.

        Returns:
            dict[str, Any]: Nested dictionary containing 'whisper', 'inference', 'gate', 'audio', and 'runtime'.
        """
        return asdict(self)

    # Save the configuration to a UTF-8 JSON file on disk
    def save_json(self, path: str | Path) -> None:
        """
        Write the serialized configuration dictionary to a formatted JSON file.

        Args:
            path (str | Path): Destination file path where the JSON configuration will be written.
        """
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    # Load and validate a KorkutConfig instance from a JSON file on disk
    @classmethod
    def load_json(cls, path: str | Path) -> KorkutConfig:
        """
        Read a JSON configuration file, verify its top-level sections, and construct a validated KorkutConfig.

        Args:
            path (str | Path): Path to the JSON configuration file to load.

        Variables:
            d (dict[str, Any]): Parsed raw JSON dictionary loaded from the file.

        Returns:
            KorkutConfig: Fully validated configuration instance.

        Raises:
            ConfigurationError: If the file cannot be read, contains invalid JSON, has unknown sections,
                or fails any sub-configuration validation check.
        """
        d: dict[str, Any]
        try:
            d = json.loads(Path(path).read_text(encoding='utf-8'))
            if set(d) - {'whisper', 'inference', 'gate', 'audio', 'runtime'}:
                raise ValueError('unknown config section')
            return cls(
                WhisperConfig(**d['whisper']),
                InferenceConfig(**d.get('inference', {})),
                GateConfig(**d.get('gate', {})),
                AudioConfig(**d.get('audio', {})),
                RuntimeConfig(**d.get('runtime', {})),
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ConfigurationError(f'invalid configuration: {exc}') from exc

