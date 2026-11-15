"""Frozen Experiment 05/06 G2-11 decoder-risk model (not a probability).

Calibration weights, standardization statistics, imputation medians, and the decision
threshold are stored in `profiles/decoder_based_b2.json` and verified by SHA-256 hash
in `korkutwhisper.profiles`.
"""

from __future__ import annotations
import math
import statistics
from collections.abc import Sequence
from typing import Any
from ..profiles import GateProfile, load_gate_profile
from ..types import GateResult

# Load the default SHA-256 verified calibration profile from profiles/decoder_based_b2.json
DEFAULT_PROFILE: GateProfile = load_gate_profile('decoder_based_b2')

# Module-level calibration attributes populated from DEFAULT_PROFILE for inspection and compatibility
PROFILE_ID: str = DEFAULT_PROFILE.profile
LOGPROB_WEIGHT: float = DEFAULT_PROFILE.logprob_weight
COMPRESSION_WEIGHT: float = DEFAULT_PROFILE.compression_weight
LOGPROB_MEAN: float = DEFAULT_PROFILE.logprob_mean
COMPRESSION_MEAN: float = DEFAULT_PROFILE.compression_mean
LOGPROB_STD: float = DEFAULT_PROFILE.logprob_std
COMPRESSION_STD: float = DEFAULT_PROFILE.compression_std
LOGPROB_MEDIAN: float = DEFAULT_PROFILE.logprob_median
COMPRESSION_MEDIAN: float = DEFAULT_PROFILE.compression_median
THRESHOLD: float = DEFAULT_PROFILE.threshold


# Frozen G2 decoder-risk gate evaluator backed by a SHA-256 verified JSON GateProfile
class G2Gate:
    """
    Evaluates stage-1 (1500-frame) decoder segments using a linear risk model over
    standardized average log-probability and compression ratio features loaded from
    a `GateProfile` configuration to decide whether a stage-2 (3000-frame) retry is needed.

    Attributes:
        profile (str): Name of the active calibration profile.
        threshold (float): Risk score threshold above or equal to which `retry_required` is True.
        verified (bool): True if the underlying JSON profile passed SHA-256 verification.
        sha256 (str): SHA-256 hex digest of the loaded JSON profile file.
    """
    profile: str = PROFILE_ID
    threshold: float = THRESHOLD

    # Initialize the G2Gate with an optional GateProfile configuration object
    def __init__(self, profile_config: GateProfile | None = None) -> None:
        """
        Configure the G2 gate evaluator using either an explicit `GateProfile` loaded from JSON
        or the default module-level `decoder_based_b2` profile constants.

        Args:
            profile_config (GateProfile | None): Optional loaded `GateProfile` instance. When None,
                uses the default `decoder_based_b2` profile attributes.
        """
        self._profile: GateProfile | None = profile_config
        if profile_config is not None:
            self.profile = profile_config.profile
            self.threshold = profile_config.threshold
            self.verified: bool = profile_config.verified
            self.sha256: str = profile_config.sha256
        else:
            self.profile = PROFILE_ID
            self.threshold = THRESHOLD
            self.verified = DEFAULT_PROFILE.verified
            self.sha256 = DEFAULT_PROFILE.sha256

    # Compute the G2 risk score and retry decision from stage-1 decoded segments
    def evaluate(self, segments: Sequence[Any]) -> GateResult:
        """
        Extract mean `avg_logprob` and `compression_ratio` across decoded segments, impute
        any missing/non-finite features with calibration medians, and compute the linear risk score.

        Args:
            segments (Sequence[Any]): Sequence of decoded segment objects from stage-1 inference.

        Variables:
            c (float): Effective compression ratio value (observed mean or imputed median).
            compression (float | None): Raw mean `compression_ratio` across segments, or None if invalid.
            l (float): Effective average log-probability value (observed mean or imputed median).
            logprob (float | None): Raw mean `avg_logprob` across segments, or None if invalid.
            missing (list[str]): List of feature names that were None/non-finite and required median imputation.
            risk (float): Computed linear combination of z-scored logprob and compression ratio features.
            p (GateProfile | None): Active explicit `GateProfile` if provided at init, else None.
            prof_id (str): Effective profile identifier string.
            lp_w (float): Effective logprob linear weight.
            cp_w (float): Effective compression ratio linear weight.
            lp_mean (float): Effective logprob standardization mean.
            cp_mean (float): Effective compression ratio standardization mean.
            lp_std (float): Effective logprob standardization standard deviation.
            cp_std (float): Effective compression ratio standardization standard deviation.
            lp_med (float): Effective logprob imputation median.
            cp_med (float): Effective compression ratio imputation median.
            thresh (float): Effective decision threshold.

        Returns:
            GateResult: Frozen result object containing the risk score, threshold, retry decision,
                raw features, and list of imputed feature names.
        """
        c: float
        compression: float | None
        l: float
        logprob: float | None
        missing: list[str]
        risk: float

        # Helper function to compute the arithmetic mean of a finite attribute across all segments
        def mean(name: str) -> float | None:
            """
            Compute the mean of attribute `name` across all segments if non-empty and all values are finite.

            Args:
                name (str): Name of the segment attribute ('avg_logprob' or 'compression_ratio').

            Variables:
                vals (list[Any]): Extracted attribute values from each segment in `segments`.

            Returns:
                float | None: Arithmetic mean of the attribute, or None if segments is empty or any value is non-finite.
            """
            vals: list[Any]
            vals = [getattr(s, name, None) for s in segments]
            return (
                statistics.mean(float(x) for x in vals)
                if vals and all(x is not None and math.isfinite(float(x)) for x in vals)
                else None
            )

        # Resolve active calibration constants from either explicit GateProfile or module attributes
        p = self._profile
        prof_id = p.profile if p is not None else PROFILE_ID
        lp_w = p.logprob_weight if p is not None else LOGPROB_WEIGHT
        cp_w = p.compression_weight if p is not None else COMPRESSION_WEIGHT
        lp_mean = p.logprob_mean if p is not None else LOGPROB_MEAN
        cp_mean = p.compression_mean if p is not None else COMPRESSION_MEAN
        lp_std = p.logprob_std if p is not None else LOGPROB_STD
        cp_std = p.compression_std if p is not None else COMPRESSION_STD
        lp_med = p.logprob_median if p is not None else LOGPROB_MEDIAN
        cp_med = p.compression_median if p is not None else COMPRESSION_MEDIAN
        thresh = p.threshold if p is not None else THRESHOLD

        # Aggregate segment-level decoder confidence and repetition metrics
        logprob = mean('avg_logprob')
        compression = mean('compression_ratio')

        # Record any missing features and fall back to calibration medians when needed
        missing = [
            name
            for name, val in [('avg_logprob', logprob), ('compression_ratio', compression)]
            if val is None
        ]
        l = lp_med if logprob is None else logprob
        c = cp_med if compression is None else compression

        # Compute the linear risk score from z-score standardized features
        risk = (
            lp_w * ((l - lp_mean) / lp_std)
            + cp_w * ((c - cp_mean) / cp_std)
        )

        return GateResult(
            prof_id,
            float(risk),
            thresh,
            bool(risk >= thresh),
            {'avg_logprob': logprob, 'compression_ratio': compression},
            missing,
        )

    # Export all active calibration constants, SHA-256 hash, and decision rules as a dictionary
    def to_dict(self) -> dict[str, Any]:
        """
        Return an inspectable dictionary containing the profile ID, feature weights,
        standardization statistics, imputation medians, threshold rule, and SHA-256 verification metadata.

        Returns:
            dict[str, Any]: Dictionary of G2 gate profile parameters and hash verification state.
        """
        if self._profile is not None:
            return self._profile.to_dict()
        return {
            'profile': PROFILE_ID,
            'logprob_weight': LOGPROB_WEIGHT,
            'compression_weight': COMPRESSION_WEIGHT,
            'logprob_mean': LOGPROB_MEAN,
            'compression_mean': COMPRESSION_MEAN,
            'logprob_std': LOGPROB_STD,
            'compression_std': COMPRESSION_STD,
            'logprob_median': LOGPROB_MEDIAN,
            'compression_median': COMPRESSION_MEDIAN,
            'threshold': THRESHOLD,
            'decision': DEFAULT_PROFILE.decision,
            'score_is_probability': DEFAULT_PROFILE.score_is_probability,
            'sha256': DEFAULT_PROFILE.sha256,
            'verified': DEFAULT_PROFILE.verified,
        }


