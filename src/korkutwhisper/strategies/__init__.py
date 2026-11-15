"""Strategy policies independent of model ownership.

Current Release Scope:
    Only short-form audio inputs up to 1500 Mel content frames (~15 seconds at 16 kHz)
    are accepted. Any audio input exceeding 1500 Mel frames is rejected immediately with
    `UnsupportedAudioLengthError`.

Future Work (Planned Long-Form Chunking Architecture):
    Audio recordings exceeding 1500 Mel content frames will be handled by a dedicated
    long-form chunking pipeline (e.g., `transcribe_long` or chunked streaming mode) that:
      1. Splits long waveforms into <=1500-Mel-frame (~15s) segments (using fixed windows
         or acoustic silence boundaries).
      2. Routes each chunk independently through the G2 strategy (running a fast 1500-frame
         stage-1 pass and selectively retrying high-risk chunks with a 3000-frame padded window).
      3. Offsets per-chunk segment `start` and `end` timestamps by the chunk's cumulative
         time position so multi-minute recordings produce accurate `.srt` and `.vtt` subtitles.
"""

from dataclasses import dataclass
from ..exceptions import UnsupportedAudioLengthError

# Maximum number of Mel spectrogram content frames permitted for a single audio input (~15 seconds)
MAX_CONTENT_FRAMES: int = 1500


# Immutable routing decision specifying initial encoder window size and gate usage
@dataclass(frozen=True)
class Route:
    """
    Describes the encoder frame window and gate activation selected for an audio input.

    Attributes:
        initial_frames (int): Number of Mel frames (1500 or 3000) to pad/pass to the stage-1 encoder.
        use_gate (bool): Whether the G2 risk gate should evaluate stage-1 segments for potential retry.
        reason (str): Diagnostic label explaining why this route was chosen.
    """
    initial_frames: int
    use_gate: bool
    reason: str


# Policy engine that selects encoder frame windows based on audio Mel length and strategy name
class Strategy:
    """
    Determines how many Mel spectrogram frames to pass to the Whisper encoder and whether
    to run the G2 gate, while strictly enforcing the 1500-Mel-frame input limit and preventing
    silent audio truncation.

    Attributes:
        name (str): Active strategy identifier ('g2', 'fixed_1500', or 'fixed_3000').
    """

    # Initialize the routing strategy with the specified strategy name
    def __init__(self, name: str) -> None:
        """
        Store the configured strategy name for subsequent routing decisions.

        Args:
            name (str): Strategy identifier ('g2', 'fixed_1500', or 'fixed_3000').
        """
        self.name: str = name

    # Determine the encoder window route for an audio clip with `actual` Mel content frames
    def route(self, actual: int) -> Route:
        """
        Select the initial encoder frame window (1500 or 3000) and gate policy for a given
        number of Mel content frames. Strictly rejects empty audio (`actual <= 0`) and any
        audio exceeding 1500 Mel content frames (`actual > 1500`).

        Future Work:
            Inputs with `actual > 1500` Mel frames are currently rejected rather than bypassed
            to a 3000-frame window. In a future release, long-form audio (>1500 Mel frames)
            will be segmented into <=1500-frame chunks before routing so each chunk remains
            within the calibrated G2 operating domain.

        Args:
            actual (int): Number of actual Mel spectrogram content frames extracted from the audio.

        Returns:
            Route: Selected routing decision with `initial_frames`, `use_gate` flag, and diagnostic `reason`.

        Raises:
            UnsupportedAudioLengthError: If `actual <= 0` or `actual > 1500` Mel content frames.
        """
        # Reject empty audio inputs with no Mel content frames
        if actual <= 0:
            raise UnsupportedAudioLengthError('audio has no Mel content frames')

        # Reject any audio exceeding the 1500 Mel-frame (~15s) calibrated short-form limit.
        # Future work: long-form audio chunking will split >1500-frame inputs into <=1500-frame
        # windows prior to routing instead of raising UnsupportedAudioLengthError here.
        if actual > MAX_CONTENT_FRAMES:
            raise UnsupportedAudioLengthError(
                f'audio length ({actual} Mel frames) exceeds the maximum supported limit of '
                f'{MAX_CONTENT_FRAMES} Mel frames (~15 seconds); long-form chunking for inputs '
                f'above {MAX_CONTENT_FRAMES} Mel frames is planned for future work'
            )

        # Route valid <=1500-frame audio according to the configured strategy
        if self.name == 'fixed_3000':
            return Route(3000, False, 'fixed_3000')
        if self.name == 'fixed_1500':
            return Route(1500, False, 'fixed_1500')
        return Route(1500, True, 'g2_eligible')


