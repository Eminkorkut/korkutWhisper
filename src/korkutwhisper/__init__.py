"""Decoder-guided CTranslate2 Whisper inference package.

Exposes the primary KorkutWhisper client, configuration dataclasses,
typed result contracts, and the public exception hierarchy.
"""

from .api import KorkutWhisper
from .audio import inspect_audio, inspect_dataset
from .cli import main as cli_main
from .config import (
    AudioConfig,
    GateConfig,
    InferenceConfig,
    KorkutConfig,
    RuntimeConfig,
    WhisperConfig,
)
from .exceptions import (
    ConfigurationError,
    InferenceError,
    InvalidAudioError,
    KorkutWhisperError,
    ModelCompatibilityError,
    ModelLoadError,
    UnsupportedAudioLengthError,
)
from .profiles import (
    GateProfile,
    VERIFIED_PROFILE_HASHES,
    load_gate_profile,
    profile_hash,
)
from .types import (
    AudioInspection,
    BatchItem,
    CSV_COLUMNS,
    DatasetInspection,
    GateResult,
    INSPECTION_CSV_COLUMNS,
    ModelInfo,
    TimingResult,
    TranscriptionResult,
    TranscriptionSegment,
    format_batch_csv,
    save_batch_csv,
    save_batch_jsonl,
)

# Current package release version
__version__ = '0.1.0'

# Public symbols exported by the top-level package
__all__ = [
    'KorkutWhisper',
    'cli_main',
    'inspect_audio',
    'inspect_dataset',
    'WhisperConfig',
    'InferenceConfig',
    'GateConfig',
    'GateProfile',
    'VERIFIED_PROFILE_HASHES',
    'load_gate_profile',
    'profile_hash',
    'AudioConfig',
    'RuntimeConfig',
    'KorkutConfig',
    'AudioInspection',
    'DatasetInspection',
    'TranscriptionResult',
    'TranscriptionSegment',
    'TimingResult',
    'GateResult',
    'ModelInfo',
    'BatchItem',
    'CSV_COLUMNS',
    'INSPECTION_CSV_COLUMNS',
    'format_batch_csv',
    'save_batch_csv',
    'save_batch_jsonl',
    'KorkutWhisperError',
    'ModelLoadError',
    'InvalidAudioError',
    'UnsupportedAudioLengthError',
    'ModelCompatibilityError',
    'InferenceError',
    'ConfigurationError',
]

