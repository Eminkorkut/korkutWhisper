"""Public exception hierarchy."""


# Base exception class for all korkutwhisper errors
class KorkutWhisperError(Exception):
    """
    Root exception for all errors raised by the korkutwhisper library.
    Allows callers to catch any library-specific failure with a single handler.
    """
    pass


# Exception raised when user or JSON configuration values are invalid
class ConfigurationError(KorkutWhisperError):
    """
    Raised when configuration parameters (device, strategy, concurrency, sample rate,
    or JSON schema) fail validation checks.
    """
    pass


# Exception raised when the local CTranslate2 model cannot be loaded
class ModelLoadError(KorkutWhisperError):
    """
    Raised when the model directory or required model files are missing, unreadable,
    or fail to initialize in the CTranslate2 / faster-whisper backend.
    """
    pass


# Exception raised when the model or runtime environment does not match requirements
class ModelCompatibilityError(KorkutWhisperError):
    """
    Raised when the faster-whisper version, Mel filterbank bin count, or unverified
    model/runtime configuration is incompatible with the active strategy or gate profile.
    """
    pass


# Exception raised when input audio files or arrays are malformed or unreadable
class InvalidAudioError(KorkutWhisperError):
    """
    Raised when an audio file is missing, lacks an audio stream, contains corrupt frames,
    or when an input NumPy array violates shape, dtype, range, or finiteness constraints.
    """
    pass


# Exception raised when audio length falls outside supported Mel frame bounds
class UnsupportedAudioLengthError(KorkutWhisperError):
    """
    Raised when the extracted Mel spectrogram has zero content frames or exceeds the
    maximum supported 1500-Mel-frame (~15 seconds) short-form input limit.
    (Long-form audio chunking for inputs above 1500 Mel frames is planned for future work.)
    """
    pass


# Exception raised when runtime feature extraction or decoding fails
class InferenceError(KorkutWhisperError):
    """
    Raised when transcription is attempted on a closed instance, or when Mel extraction,
    encoder window shape verification, or CTranslate2 decoding fails at runtime.
    """
    pass

