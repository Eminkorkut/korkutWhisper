"""Versioned, SHA-256 verified inference profiles loaded from JSON configuration files."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from ..exceptions import ConfigurationError

# Directory containing built-in JSON gate profile files
PROFILES_DIR: Path = Path(__file__).resolve().parent

# Default verified profile identifier
PROFILE_ID: str = 'decoder_based_b2'

# Expected SHA-256 hex digests (over LF-normalized UTF-8 bytes) for certified built-in profiles
VERIFIED_PROFILE_HASHES: dict[str, str] = {
    'decoder_based_b2': '70e68351003baabf12a6371c32c155aaa0d44f4eb535e141e189aac0c9d23e37',
}


# Compute the line-ending-normalized SHA-256 hex digest of a JSON profile file
def profile_hash(path: str | Path) -> str:
    """
    Read a gate profile JSON file from disk, normalize CRLF (`\\r\\n`) line endings to LF (`\\n`)
    so the digest is identical across Windows and POSIX platforms, and compute its SHA-256 hex digest.

    Args:
        path (str | Path): Filesystem path to the JSON profile file.

    Variables:
        raw (bytes): Raw binary contents of the profile file with CRLF normalized to LF.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest string.

    Raises:
        ConfigurationError: If the profile file does not exist or cannot be read.
    """
    raw: bytes
    try:
        raw = Path(path).read_bytes().replace(b'\r\n', b'\n')
    except OSError as exc:
        raise ConfigurationError(f'cannot read gate profile file {path}: {exc}') from exc
    return hashlib.sha256(raw).hexdigest()


# Immutable container holding all calibration parameters and hash verification status for a G2 profile
@dataclass(frozen=True)
class GateProfile:
    """
    Stores the linear risk weights, standardization statistics, imputation medians,
    decision threshold, and cryptographic SHA-256 verification status loaded from a profile JSON file.

    Attributes:
        profile (str): Unique profile identifier (e.g., `'decoder_based_b2'`).
        logprob_weight (float): Linear weight applied to standardized `avg_logprob`.
        compression_weight (float): Linear weight applied to standardized `compression_ratio`.
        logprob_mean (float): Calibration mean for `avg_logprob`.
        compression_mean (float): Calibration mean for `compression_ratio`.
        logprob_std (float): Calibration standard deviation for `avg_logprob` (must be > 0).
        compression_std (float): Calibration standard deviation for `compression_ratio` (must be > 0).
        logprob_median (float): Imputation median when `avg_logprob` is missing or non-finite.
        compression_median (float): Imputation median when `compression_ratio` is missing or non-finite.
        threshold (float): Risk score threshold above or equal to which a stage-2 retry is triggered.
        decision (str): Human-readable decision rule description (`'risk_score >= threshold'`).
        score_is_probability (bool): Indicates whether the risk score is a calibrated probability (False).
        sha256 (str): Computed SHA-256 hex digest of the loaded JSON profile file.
        verified (bool): True if `sha256` matches the certified entry in `VERIFIED_PROFILE_HASHES`.
    """
    profile: str
    logprob_weight: float
    compression_weight: float
    logprob_mean: float
    compression_mean: float
    logprob_std: float
    compression_std: float
    logprob_median: float
    compression_median: float
    threshold: float
    decision: str = 'risk_score >= threshold'
    score_is_probability: bool = False
    sha256: str = ''
    verified: bool = False

    # Validate numeric finiteness and positive standard deviations on initialization
    def __post_init__(self) -> None:
        """
        Ensure all numerical calibration parameters are finite floats and standard deviations are positive.

        Raises:
            ConfigurationError: If any numeric parameter is non-finite or standard deviations are <= 0.
        """
        numeric_fields = (
            self.logprob_weight,
            self.compression_weight,
            self.logprob_mean,
            self.compression_mean,
            self.logprob_std,
            self.compression_std,
            self.logprob_median,
            self.compression_median,
            self.threshold,
        )
        if not self.profile or not isinstance(self.profile, str):
            raise ConfigurationError('profile identifier must be a non-empty string')
        if any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v))
            for v in numeric_fields
        ):
            raise ConfigurationError('all gate profile numeric parameters must be finite numbers')
        if float(self.logprob_std) <= 0.0 or float(self.compression_std) <= 0.0:
            raise ConfigurationError('gate profile standard deviations must be strictly positive')

    # Export profile parameters and verification metadata as a dictionary
    def to_dict(self) -> dict[str, Any]:
        """
        Convert this `GateProfile` into an inspectable dictionary including its SHA-256 digest
        and verification flag.

        Returns:
            dict[str, Any]: Dictionary of profile constants and verification metadata.
        """
        return {
            'profile': self.profile,
            'logprob_weight': self.logprob_weight,
            'compression_weight': self.compression_weight,
            'logprob_mean': self.logprob_mean,
            'compression_mean': self.compression_mean,
            'logprob_std': self.logprob_std,
            'compression_std': self.compression_std,
            'logprob_median': self.logprob_median,
            'compression_median': self.compression_median,
            'threshold': self.threshold,
            'decision': self.decision,
            'score_is_probability': self.score_is_probability,
            'sha256': self.sha256,
            'verified': self.verified,
        }


# Load a GateProfile from a built-in profile name or custom JSON file path and verify its SHA-256 hash
def load_gate_profile(
    profile: str = PROFILE_ID,
    profile_path: str | Path | None = None,
) -> GateProfile:
    """
    Load a G2 calibration profile from a JSON configuration file (either from the built-in
    `profiles/<profile>.json` directory or an explicit `profile_path`), validate its schema,
    and verify its SHA-256 digest against `VERIFIED_PROFILE_HASHES`.

    Args:
        profile (str): Profile name identifier (defaults to `'decoder_based_b2'`).
        profile_path (str | Path | None): Optional path to a custom external JSON profile file.

    Variables:
        target (Path): Resolved filesystem path to the JSON profile file.
        digest (str): Computed LF-normalized SHA-256 hex digest of `target`.
        data (dict[str, Any]): Parsed JSON dictionary from `target`.
        expected_hash (str | None): Certified SHA-256 hash from `VERIFIED_PROFILE_HASHES`, if any.
        is_verified (bool): True if loaded from a certified profile whose SHA-256 matches `expected_hash`.

    Returns:
        GateProfile: Loaded, validated, and hash-checked profile configuration object.

    Raises:
        ConfigurationError: If the profile file is missing, unreadable, malformed, or fails validation.
    """
    target: Path
    digest: str
    data: dict[str, Any]
    expected_hash: str | None
    is_verified: bool

    if profile_path is not None:
        target = Path(profile_path)
    else:
        target = PROFILES_DIR / f'{profile}.json'

    if not target.is_file():
        raise ConfigurationError(f'gate profile JSON not found: {target}')

    digest = profile_hash(target)
    try:
        data = json.loads(target.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('gate profile JSON root must be an object')
        required_keys = {
            'profile',
            'logprob_weight',
            'compression_weight',
            'logprob_mean',
            'compression_mean',
            'logprob_std',
            'compression_std',
            'logprob_median',
            'compression_median',
            'threshold',
        }
        missing = required_keys - set(data)
        if missing:
            raise ValueError(f'missing required profile keys: {sorted(missing)}')

        profile_name = str(data['profile'])
        expected_hash = VERIFIED_PROFILE_HASHES.get(profile_name)
        is_verified = bool(
            profile_path is None
            and expected_hash is not None
            and digest == expected_hash
        )

        return GateProfile(
            profile=profile_name,
            logprob_weight=float(data['logprob_weight']),
            compression_weight=float(data['compression_weight']),
            logprob_mean=float(data['logprob_mean']),
            compression_mean=float(data['compression_mean']),
            logprob_std=float(data['logprob_std']),
            compression_std=float(data['compression_std']),
            logprob_median=float(data['logprob_median']),
            compression_median=float(data['compression_median']),
            threshold=float(data['threshold']),
            decision=str(data.get('decision', 'risk_score >= threshold')),
            score_is_probability=bool(data.get('score_is_probability', False)),
            sha256=digest,
            verified=is_verified,
        )
    except ConfigurationError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ConfigurationError(f'invalid gate profile {target}: {exc}') from exc


# Lazily expose G2Gate on korkutwhisper.profiles to avoid circular imports with korkutwhisper.gates.g2
def __getattr__(name: str) -> Any:
    """
    Lazily import and return `G2Gate` from `korkutwhisper.gates.g2` when accessed via
    `korkutwhisper.profiles.G2Gate`.

    Args:
        name (str): Attribute name requested on the `korkutwhisper.profiles` module.

    Returns:
        Any: The `G2Gate` class when `name == 'G2Gate'`.

    Raises:
        AttributeError: If `name` is not a recognized module attribute.
    """
    if name == 'G2Gate':
        from ..gates.g2 import G2Gate
        return G2Gate
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Exported profile symbols
__all__ = [
    'G2Gate',
    'GateProfile',
    'PROFILE_ID',
    'PROFILES_DIR',
    'VERIFIED_PROFILE_HASHES',
    'load_gate_profile',
    'profile_hash',
]


