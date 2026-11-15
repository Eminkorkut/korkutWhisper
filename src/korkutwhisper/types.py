"""Typed result contracts and output formatters. Times are wall-clock milliseconds."""

from __future__ import annotations
from collections.abc import Sequence
import csv
from dataclasses import asdict, dataclass, field
import io
import json
from pathlib import Path
from typing import Any
from .exceptions import ConfigurationError

# Ordered column names used when exporting TranscriptionResult or BatchItem rows to CSV
CSV_COLUMNS: tuple[str, ...] = (
    'index',
    'status',
    'audio_id',
    'text',
    'first_stage_text',
    'language',
    'duration',
    'strategy',
    'retry_performed',
    'final_stage',
    'risk_score',
    'risk_threshold',
    'retry_required',
    'avg_logprob',
    'compression_ratio',
    'audio_decode_ms',
    'preprocessing_ms',
    'encoder_stage1_ms',
    'decoder_stage1_ms',
    'encoder_stage2_ms',
    'decoder_stage2_ms',
    'gate_ms',
    'total_ms',
    'rtf',
    'error_type',
    'error_message',
)

# Ordered column names used when exporting AudioInspection or DatasetInspection rows to CSV
INSPECTION_CSV_COLUMNS: tuple[str, ...] = (
    'index',
    'audio_id',
    'source',
    'valid',
    'status',
    'duration',
    'num_samples',
    'actual_mel_frames',
    'max_allowed_frames',
    'peak_amplitude',
    'rms_amplitude',
    'error_type',
    'error_message',
)


# Format a timestamp in seconds into SRT ('HH:MM:SS,mmm') or WebVTT ('HH:MM:SS.mmm') notation
def _format_timestamp(seconds: float, *, separator: str = ',') -> str:
    """
    Convert a non-negative duration in seconds into a zero-padded subtitle timestamp string.

    Args:
        seconds (float): Timestamp in seconds (clamped to >= 0.0).
        separator (str): Millisecond separator character (',' for SRT, '.' for WebVTT).

    Variables:
        total_ms (int): Total rounded milliseconds derived from `seconds`.
        hours (int): Whole hours component of the timestamp.
        rem (int): Remaining milliseconds after subtracting whole hours.
        minutes (int): Whole minutes component (0–59).
        secs (int): Whole seconds component (0–59).
        millis (int): Milliseconds component (0–999).

    Returns:
        str: Formatted timestamp string `'HH:MM:SS,mmm'` or `'HH:MM:SS.mmm'`.
    """
    total_ms: int
    hours: int
    rem: int
    minutes: int
    secs: int
    millis: int

    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1_000)
    return f'{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}'


# Base mixin class providing dictionary and JSON serialization for dataclass results
class JsonResult:
    """
    Mixin inherited by all public result dataclasses to provide uniform
    dictionary and JSON serialization methods.
    """

    # Convert the dataclass instance and nested dataclasses into a plain dictionary
    def to_dict(self) -> dict[str, Any]:
        """
        Recursively serialize this dataclass instance into a standard Python dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of all dataclass fields.
        """
        return asdict(self)

    # Serialize the dataclass instance into a formatted UTF-8 JSON string
    def to_json(self, *, indent: int | None = 2) -> str:
        """
        Convert this result dataclass into a JSON string with Turkish/Unicode characters preserved.

        Args:
            indent (int | None): Indentation spaces for pretty-printing, or None for compact single-line JSON.

        Returns:
            str: Serialized JSON string.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    # Write the JSON representation of this result dataclass to a file on disk
    def save_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """
        Serialize this result object as JSON and write it to `path` using UTF-8 encoding.

        Args:
            path (str | Path): Destination filesystem path for the JSON file.
            indent (int | None): Indentation spaces for pretty-printing. Defaults to 2.

        Variables:
            target (Path): Resolved `Path` object where the JSON content is written.

        Returns:
            Path: The destination path written to disk.
        """
        target: Path
        target = Path(path)
        target.write_text(self.to_json(indent=indent) + '\n', encoding='utf-8')
        return target


# Immutable container representing a single decoded transcription segment
@dataclass(frozen=True)
class TranscriptionSegment(JsonResult):
    """
    Represents an individual decoded text segment and its associated decoder metrics.

    Attributes:
        text (str): Stripped transcribed text produced by the decoder for this segment.
        start (float | None): Segment start timestamp in seconds, or None when timestamps are disabled.
        end (float | None): Segment end timestamp in seconds, or None when timestamps are disabled.
        avg_logprob (float | None): Mean token log-probability for the segment, used by the G2 gate.
        compression_ratio (float | None): Text compression ratio for the segment, used to detect repetition.
        tokens (list[int] | None): Sequence of token IDs generated for this segment.
    """
    text: str
    start: float | None = None
    end: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    tokens: list[int] | None = None


# Immutable container holding the G2 decoder-risk gate evaluation output
@dataclass(frozen=True)
class GateResult(JsonResult):
    """
    Stores the evaluation metrics and decision produced by the G2 risk gate after stage-1 decoding.

    Attributes:
        name (str): Identifier of the gate profile used for evaluation.
        risk_score (float): Linear risk score computed from standardized decoder features (not a probability).
        threshold (float): Decision boundary above or equal to which a stage-2 retry is triggered.
        retry_required (bool): True if risk_score >= threshold, indicating a 3000-frame retry is needed.
        features (dict[str, float | None]): Raw observed feature values ('avg_logprob', 'compression_ratio').
        imputed_features (list[str]): Names of features that were missing/non-finite and replaced with medians.
    """
    name: str
    risk_score: float
    threshold: float
    retry_required: bool
    features: dict[str, float | None]
    imputed_features: list[str] = field(default_factory=list)


# Immutable container capturing wall-clock execution timings across pipeline stages
@dataclass(frozen=True)
class TimingResult(JsonResult):
    """
    Records host wall-clock durations (in milliseconds) for each stage of the transcription pipeline.

    Attributes:
        audio_decode_ms (float): Time spent decoding the audio file or preparing the input NumPy array.
        preprocessing_ms (float): Time spent computing the Mel spectrogram features.
        encoder_stage1_ms (float): Time spent in the Whisper encoder during the initial pass.
        decoder_stage1_ms (float): Time spent in the Whisper decoder during the initial pass.
        encoder_stage2_ms (float | None): Time spent in the encoder during the retry pass, or None if not retried.
        decoder_stage2_ms (float | None): Time spent in the decoder during the retry pass, or None if not retried.
        gate_ms (float): Time spent evaluating the G2 gate decision model.
        total_ms (float): Total end-to-end wall-clock time for the transcription call in milliseconds.
        rtf (float): Real-Time Factor computed as total_ms divided by audio duration in milliseconds.
    """
    audio_decode_ms: float
    preprocessing_ms: float
    encoder_stage1_ms: float
    decoder_stage1_ms: float
    encoder_stage2_ms: float | None
    decoder_stage2_ms: float | None
    gate_ms: float
    total_ms: float
    rtf: float


# Immutable container representing the complete output of a transcription request
@dataclass(frozen=True)
class TranscriptionResult(JsonResult):
    """
    Complete output returned by KorkutWhisper.transcribe() and KorkutWhisper.transcribe_array().

    Attributes:
        text (str): Full joined transcription text from the final selected decoding stage.
        language (str): Language code used for decoding (e.g., 'tr').
        duration (float): Duration of the input audio in seconds (based on 16 kHz sample count).
        strategy (str): Name of the routing strategy applied ('g2', 'fixed_1500', or 'fixed_3000').
        retry_performed (bool): Whether a second-stage 3000-frame retry pass was executed.
        final_stage (int): The decoding stage that produced the final result (1 for initial, 2 for retry).
        segments (list[TranscriptionSegment]): List of decoded segments from the final stage.
        gate (GateResult | None): G2 gate evaluation details if the gate was used, otherwise None.
        timing (TimingResult): Breakdown of wall-clock execution timings and Real-Time Factor.
        metadata (dict[str, Any]): Diagnostic metadata including routing reason, frame counts, and stage-1 text.
    """
    text: str
    language: str
    duration: float
    strategy: str
    retry_performed: bool
    final_stage: int
    segments: list[TranscriptionSegment]
    gate: GateResult | None
    timing: TimingResult
    metadata: dict[str, Any]

    # Return the plain transcribed text string
    def to_text(self) -> str:
        """
        Return the clean transcribed text without extra metadata.

        Returns:
            str: Transcribed text (`self.text`).
        """
        return self.text

    # Build a human-readable multi-line diagnostic report of the transcription result
    def to_report(self) -> str:
        """
        Format the transcription result, routing decision, G2 gate evaluation, and stage timings
        into a structured human-readable summary report suitable for console output or log files.

        Variables:
            audio_id (Any): Identifier of the transcribed audio from `self.metadata`.
            routing_reason (Any): Routing reason string from `self.metadata`.
            frames (Any): List of encoder frame window sizes invoked during inference.
            lines (list[str]): Ordered report lines joined with newlines at the end.

        Returns:
            str: Multi-line human-readable diagnostic report string.
        """
        audio_id: Any
        routing_reason: Any
        frames: Any
        lines: list[str]

        audio_id = self.metadata.get('audio_id', 'unknown')
        routing_reason = self.metadata.get('routing_reason', 'unknown')
        frames = self.metadata.get('encoder_input_frames', [])

        lines = [
            f'=== KorkutWhisper Transcription Report ({audio_id}) ===',
            f'Language / Duration : {self.language} | {self.duration:.2f}s',
            f'Strategy / Routing  : {self.strategy} ({routing_reason}) | frames={frames}',
            f'Final Stage / Retry : stage {self.final_stage} | retry_performed={self.retry_performed}',
        ]

        if self.gate is not None:
            lines.append(
                f'G2 Gate Evaluation  : risk={self.gate.risk_score:.6f} '
                f'(threshold={self.gate.threshold:.6f}, retry_required={self.gate.retry_required})'
            )
            lines.append(
                f'G2 Decoder Features : avg_logprob={self.gate.features.get("avg_logprob")} | '
                f'compression_ratio={self.gate.features.get("compression_ratio")}'
            )
        else:
            lines.append('G2 Gate Evaluation  : bypassed / not used')

        lines.append(
            f'Timing Breakdown    : total={self.timing.total_ms:.2f}ms (RTF={self.timing.rtf:.4f}) | '
            f'decode={self.timing.audio_decode_ms:.2f}ms | mel={self.timing.preprocessing_ms:.2f}ms | '
            f'enc1={self.timing.encoder_stage1_ms:.2f}ms | dec1={self.timing.decoder_stage1_ms:.2f}ms'
            + (
                f' | enc2={self.timing.encoder_stage2_ms:.2f}ms | dec2={self.timing.decoder_stage2_ms:.2f}ms'
                if self.timing.encoder_stage2_ms is not None and self.timing.decoder_stage2_ms is not None
                else ''
            )
        )
        lines.append(f'Text                : {self.text}')
        return '\n'.join(lines)

    # Flatten nested transcription, gate, timing, and metadata fields into a single-level dictionary
    def to_csv_row(self, *, index: int = 0) -> dict[str, Any]:
        """
        Convert this `TranscriptionResult` into a flat single-level dictionary matching `CSV_COLUMNS`
        for tabular export and spreadsheet analysis.

        Args:
            index (int): Zero-based row index (used when exporting single or batch results). Defaults to 0.

        Returns:
            dict[str, Any]: Flat dictionary mapping each column in `CSV_COLUMNS` to its scalar value.
        """
        return {
            'index': index,
            'status': 'ok',
            'audio_id': self.metadata.get('audio_id', ''),
            'text': self.text,
            'first_stage_text': self.metadata.get('first_stage_text', ''),
            'language': self.language,
            'duration': self.duration,
            'strategy': self.strategy,
            'retry_performed': self.retry_performed,
            'final_stage': self.final_stage,
            'risk_score': self.gate.risk_score if self.gate is not None else None,
            'risk_threshold': self.gate.threshold if self.gate is not None else None,
            'retry_required': self.gate.retry_required if self.gate is not None else None,
            'avg_logprob': self.gate.features.get('avg_logprob') if self.gate is not None else None,
            'compression_ratio': self.gate.features.get('compression_ratio') if self.gate is not None else None,
            'audio_decode_ms': self.timing.audio_decode_ms,
            'preprocessing_ms': self.timing.preprocessing_ms,
            'encoder_stage1_ms': self.timing.encoder_stage1_ms,
            'decoder_stage1_ms': self.timing.decoder_stage1_ms,
            'encoder_stage2_ms': self.timing.encoder_stage2_ms,
            'decoder_stage2_ms': self.timing.decoder_stage2_ms,
            'gate_ms': self.timing.gate_ms,
            'total_ms': self.timing.total_ms,
            'rtf': self.timing.rtf,
            'error_type': None,
            'error_message': None,
        }

    # Serialize this transcription result as a CSV string
    def to_csv(self, *, include_header: bool = True) -> str:
        """
        Format this `TranscriptionResult` as a RFC-4180 compliant CSV string.

        Args:
            include_header (bool): Whether to include the column header line before the data row. Defaults to True.

        Variables:
            buf (io.StringIO): In-memory string buffer used by `csv.DictWriter`.
            writer (csv.DictWriter): CSV writer configured with `CSV_COLUMNS`.

        Returns:
            str: CSV string containing the flattened transcription metrics and text.
        """
        buf: io.StringIO
        writer: csv.DictWriter[str]

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator='\n')
        if include_header:
            writer.writeheader()
        writer.writerow(self.to_csv_row(index=0))
        return buf.getvalue()

    # Format the transcription into SubRip (.srt) subtitle format
    def to_srt(self) -> str:
        """
        Render the transcription into SubRip (`.srt`) subtitle cues.
        Uses segment-level `start` and `end` timestamps when present, or falls back to a single
        cue spanning `00:00:00,000 --> duration` when segment timestamps are `None`.

        Variables:
            blocks (list[str]): Formatted SRT cue blocks.
            idx (int): 1-based subtitle sequence number.
            seg (TranscriptionSegment): Current segment being formatted.
            start_ts (str): Formatted start timestamp (`HH:MM:SS,mmm`).
            end_ts (str): Formatted end timestamp (`HH:MM:SS,mmm`).

        Returns:
            str: SubRip (`.srt`) formatted string.
        """
        blocks: list[str]
        idx: int
        seg: TranscriptionSegment
        start_ts: str
        end_ts: str

        blocks = []
        for idx, seg in enumerate((s for s in self.segments if s.text), start=1):
            start_ts = _format_timestamp(seg.start if seg.start is not None else 0.0, separator=',')
            end_ts = _format_timestamp(seg.end if seg.end is not None else self.duration, separator=',')
            blocks.append(f'{idx}\n{start_ts} --> {end_ts}\n{seg.text}')

        if not blocks:
            start_ts = _format_timestamp(0.0, separator=',')
            end_ts = _format_timestamp(self.duration, separator=',')
            blocks.append(f'1\n{start_ts} --> {end_ts}\n{self.text}')

        return '\n\n'.join(blocks) + '\n'

    # Format the transcription into WebVTT (.vtt) subtitle format
    def to_vtt(self) -> str:
        """
        Render the transcription into WebVTT (`.vtt`) subtitle format starting with the `WEBVTT` header.
        Uses segment-level `start` and `end` timestamps when present, or falls back to a single
        cue spanning `00:00:00.000 --> duration` when segment timestamps are `None`.

        Variables:
            blocks (list[str]): Header and formatted WebVTT cue blocks.
            seg (TranscriptionSegment): Current segment being formatted.
            start_ts (str): Formatted start timestamp (`HH:MM:SS.mmm`).
            end_ts (str): Formatted end timestamp (`HH:MM:SS.mmm`).

        Returns:
            str: WebVTT (`.vtt`) formatted string.
        """
        blocks: list[str]
        seg: TranscriptionSegment
        start_ts: str
        end_ts: str

        blocks = ['WEBVTT']
        for seg in (s for s in self.segments if s.text):
            start_ts = _format_timestamp(seg.start if seg.start is not None else 0.0, separator='.')
            end_ts = _format_timestamp(seg.end if seg.end is not None else self.duration, separator='.')
            blocks.append(f'{start_ts} --> {end_ts}\n{seg.text}')

        if len(blocks) == 1:
            start_ts = _format_timestamp(0.0, separator='.')
            end_ts = _format_timestamp(self.duration, separator='.')
            blocks.append(f'{start_ts} --> {end_ts}\n{self.text}')

        return '\n\n'.join(blocks) + '\n'

    # Render the result into any supported output format by name
    def format(self, fmt: str = 'text') -> str:
        """
        Serialize this `TranscriptionResult` into the requested output format.

        Supported formats:
            - `'text'` or `'txt'`: Plain transcribed text (`to_text()`).
            - `'json'`: Pretty-printed JSON string (`to_json()`).
            - `'csv'`: CSV header + row string (`to_csv()`).
            - `'report'`: Multi-line human-readable summary report (`to_report()`).
            - `'srt'`: SubRip subtitle format (`to_srt()`).
            - `'vtt'`: WebVTT subtitle format (`to_vtt()`).

        Args:
            fmt (str): Output format identifier (case-insensitive). Defaults to `'text'`.

        Variables:
            key (str): Lowercased and stripped format name.

        Returns:
            str: Formatted output string.

        Raises:
            ConfigurationError: If `fmt` is not one of the supported output format names.
        """
        key: str
        key = str(fmt).strip().lower().lstrip('.')
        if key in ('text', 'txt'):
            return self.to_text()
        if key == 'json':
            return self.to_json(indent=2)
        if key == 'csv':
            return self.to_csv(include_header=True)
        if key == 'report':
            return self.to_report()
        if key == 'srt':
            return self.to_srt()
        if key == 'vtt':
            return self.to_vtt()
        raise ConfigurationError(
            f"unsupported output format '{fmt}'; choose from: text, txt, json, csv, report, srt, vtt"
        )

    # Save the transcription result to disk in the specified or extension-inferred format
    def save(self, path: str | Path, fmt: str | None = None) -> Path:
        """
        Write this `TranscriptionResult` to `path` using either an explicit format `fmt`
        or the format inferred from `path`'s file extension (`.txt`, `.json`, `.csv`, `.report`, `.srt`, `.vtt`).

        Args:
            path (str | Path): Destination file path.
            fmt (str | None): Optional explicit format name overriding file extension inference.

        Variables:
            target (Path): Resolved destination `Path` object.
            chosen (str): Effective format name determined from `fmt` or `target.suffix`.
            content (str): Formatted text content to write to `target`.

        Returns:
            Path: The destination file path written to disk.

        Raises:
            ConfigurationError: If the format cannot be inferred or is unsupported.
        """
        target: Path
        chosen: str
        content: str

        target = Path(path)
        chosen = fmt if fmt is not None else target.suffix.lstrip('.')
        if not chosen:
            chosen = 'text'
        content = self.format(chosen)
        if not content.endswith('\n'):
            content += '\n'
        target.write_text(content, encoding='utf-8')
        return target


# Immutable container describing the loaded model and runtime validation status
@dataclass(frozen=True)
class ModelInfo(JsonResult):
    """
    Metadata describing the loaded CTranslate2 Whisper model and its verification state.

    Attributes:
        model_path (str): Resolved filesystem path to the local model directory.
        model_family (str): Architecture family name (e.g., 'Whisper').
        compute_type (str): Quantization or floating-point compute type (e.g., 'int8').
        device (str): Execution device ('cuda', 'cpu', or 'auto').
        expected_mel_bins (int): Number of Mel frequency bins required by the model (128).
        language (str): Configured target language code.
        task (str): Configured Whisper task ('transcribe' or 'translate').
        library_version (str): Version string of the korkutwhisper library.
        active_gate_profile (str | None): Active G2 profile ID when strategy is 'g2', otherwise None.
        model_fingerprint (str): SHA-256 based fingerprint of the local model.bin file.
        validated_model (bool): True if the model files, runtime versions, and settings match the verified profile.
    """
    model_path: str
    model_family: str
    compute_type: str
    device: str
    expected_mel_bins: int
    language: str
    task: str
    library_version: str
    active_gate_profile: str | None
    model_fingerprint: str
    validated_model: bool


# Immutable container representing a single item outcome in batch transcription
@dataclass(frozen=True)
class BatchItem(JsonResult):
    """
    Wraps the result or captured exception of an individual audio item in transcribe_many().

    Attributes:
        index (int): Zero-based position of the audio item in the input iterable.
        result (TranscriptionResult | None): TranscriptionResult on success, or None if an error occurred.
        error_type (str | None): Exception class name if transcription failed, otherwise None.
        error_message (str | None): Exception message string if transcription failed, otherwise None.
    """
    index: int
    result: TranscriptionResult | None
    error_type: str | None = None
    error_message: str | None = None

    # Convert this batch item (success or error) into a flat dictionary matching CSV_COLUMNS
    def to_csv_row(self) -> dict[str, Any]:
        """
        Convert this `BatchItem` into a flat single-level dictionary matching `CSV_COLUMNS`.
        If `self.result` is present, delegates to `self.result.to_csv_row(index=self.index)`;
        otherwise returns an error row with `status='error'` and populated error fields.

        Variables:
            row (dict[str, Any]): Flat dictionary initialized with None values for failed items.

        Returns:
            dict[str, Any]: Flat dictionary suitable for `csv.DictWriter`.
        """
        row: dict[str, Any]
        if self.result is not None:
            return self.result.to_csv_row(index=self.index)
        row = {col: None for col in CSV_COLUMNS}
        row['index'] = self.index
        row['status'] = 'error'
        row['error_type'] = self.error_type
        row['error_message'] = self.error_message
        return row


# Format a sequence of BatchItem or TranscriptionResult objects as a complete CSV string
def format_batch_csv(items: Sequence[BatchItem | TranscriptionResult]) -> str:
    """
    Convert a sequence of `BatchItem` or `TranscriptionResult` instances into a multi-row
    CSV string with headers matching `CSV_COLUMNS`.

    Args:
        items (Sequence[BatchItem | TranscriptionResult]): Sequence of batch outcomes or transcription results.

    Variables:
        buf (io.StringIO): In-memory string buffer accumulating CSV output.
        writer (csv.DictWriter): CSV writer configured with `CSV_COLUMNS`.
        idx (int): Fallback zero-based index when an element is a bare `TranscriptionResult`.
        item (BatchItem | TranscriptionResult): Current element being serialized.

    Returns:
        str: Complete CSV string including header row and one row per item.
    """
    buf: io.StringIO
    writer: csv.DictWriter[str]
    idx: int
    item: BatchItem | TranscriptionResult

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), lineterminator='\n')
    writer.writeheader()
    for idx, item in enumerate(items):
        if isinstance(item, BatchItem):
            writer.writerow(item.to_csv_row())
        elif isinstance(item, TranscriptionResult):
            writer.writerow(item.to_csv_row(index=idx))
        else:
            raise ConfigurationError('items must contain BatchItem or TranscriptionResult instances')
    return buf.getvalue()


# Save a sequence of BatchItem or TranscriptionResult objects to a UTF-8 CSV file
def save_batch_csv(items: Sequence[BatchItem | TranscriptionResult], path: str | Path) -> Path:
    """
    Write a sequence of `BatchItem` or `TranscriptionResult` objects to a CSV file on disk.

    Args:
        items (Sequence[BatchItem | TranscriptionResult]): Sequence of batch outcomes or transcription results.
        path (str | Path): Destination CSV file path.

    Variables:
        target (Path): Resolved destination `Path` object.

    Returns:
        Path: The destination CSV path written to disk.
    """
    target: Path
    target = Path(path)
    target.write_text(format_batch_csv(items), encoding='utf-8')
    return target


# Save a sequence of BatchItem or TranscriptionResult objects to a UTF-8 JSON Lines (.jsonl) file
def save_batch_jsonl(items: Sequence[BatchItem | TranscriptionResult], path: str | Path) -> Path:
    """
    Write a sequence of `BatchItem` or `TranscriptionResult` objects to a JSON Lines (`.jsonl`)
    file where each line is a compact JSON object.

    Args:
        items (Sequence[BatchItem | TranscriptionResult]): Sequence of batch outcomes or transcription results.
        path (str | Path): Destination `.jsonl` file path.

    Variables:
        target (Path): Resolved destination `Path` object.
        lines (list[str]): Compact single-line JSON strings for each item.
        item (BatchItem | TranscriptionResult): Current element being serialized.

    Returns:
        Path: The destination `.jsonl` path written to disk.

    Raises:
        ConfigurationError: If any element in `items` is not a `BatchItem` or `TranscriptionResult`.
    """
    target: Path
    lines: list[str]
    item: BatchItem | TranscriptionResult

    target = Path(path)
    lines = []
    for item in items:
        if not isinstance(item, (BatchItem, TranscriptionResult)):
            raise ConfigurationError('items must contain BatchItem or TranscriptionResult instances')
        lines.append(item.to_json(indent=None))
    target.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    return target


# Immutable container holding pre-inference inspection metrics for a single audio input
@dataclass(frozen=True)
class AudioInspection(JsonResult):
    """
    Represents the pre-inference validation and Mel-frame analysis of a single audio file
    or in-memory NumPy array, computed without loading or running the GPU Whisper model.

    Attributes:
        audio_id (str): Identifier for the audio item (file stem for paths or `'array'`).
        source (str | None): Resolved filesystem path string if inspected from a file, otherwise None.
        valid (bool): True if the audio decoded cleanly and `1 <= actual_mel_frames <= max_allowed_frames`.
        status (str): Diagnostic status code (`'eligible'`, `'exceeds_1500_frames'`,
            `'empty_mel_frames'`, or `'invalid_audio'`).
        duration (float): Audio duration in seconds at 16 kHz (`0.0` if decoding failed).
        num_samples (int): Total 16 kHz mono sample count (`0` if decoding failed).
        actual_mel_frames (int): Computed Mel spectrogram content frame count (`num_samples // 160`).
        max_allowed_frames (int): Maximum supported Mel content frames (`1500`).
        peak_amplitude (float | None): Maximum absolute normalized sample amplitude, or None if invalid.
        rms_amplitude (float | None): Root-mean-square normalized signal amplitude, or None if invalid.
        error_type (str | None): Exception type name if `valid` is False, otherwise None.
        error_message (str | None): Detailed English validation failure message if `valid` is False, otherwise None.
    """
    audio_id: str
    source: str | None
    valid: bool
    status: str
    duration: float
    num_samples: int
    actual_mel_frames: int
    max_allowed_frames: int
    peak_amplitude: float | None = None
    rms_amplitude: float | None = None
    error_type: str | None = None
    error_message: str | None = None

    # Format a human-readable diagnostic summary for this single audio inspection
    def to_report(self) -> str:
        """
        Render a concise multi-line human-readable inspection report for this audio item.

        Variables:
            lines (list[str]): Report lines describing eligibility, Mel frame count, duration, and signal levels.

        Returns:
            str: Multi-line English inspection summary string.
        """
        lines: list[str]
        lines = [
            f'=== Audio Inspection ({self.audio_id}) ===',
            f'Source              : {self.source if self.source is not None else "<in-memory array>"}',
            f'Status / Valid      : {self.status} (valid={self.valid})',
            f'Duration / Samples  : {self.duration:.2f}s ({self.num_samples} samples @ 16 kHz)',
            f'Mel Content Frames  : {self.actual_mel_frames} / {self.max_allowed_frames} max frames',
        ]
        if self.peak_amplitude is not None and self.rms_amplitude is not None:
            lines.append(
                f'Signal Amplitude    : peak={self.peak_amplitude:.4f} | rms={self.rms_amplitude:.4f}'
            )
        if self.error_type is not None or self.error_message is not None:
            lines.append(f'Validation Issue    : {self.error_type}: {self.error_message}')
        return '\n'.join(lines)

    # Convert this AudioInspection into a flat dictionary matching INSPECTION_CSV_COLUMNS
    def to_csv_row(self, *, index: int = 0) -> dict[str, Any]:
        """
        Convert this `AudioInspection` into a flat single-level dictionary for CSV export.

        Args:
            index (int): Zero-based item index in the inspection run. Defaults to 0.

        Returns:
            dict[str, Any]: Flat dictionary matching `INSPECTION_CSV_COLUMNS`.
        """
        return {
            'index': index,
            'audio_id': self.audio_id,
            'source': self.source,
            'valid': self.valid,
            'status': self.status,
            'duration': self.duration,
            'num_samples': self.num_samples,
            'actual_mel_frames': self.actual_mel_frames,
            'max_allowed_frames': self.max_allowed_frames,
            'peak_amplitude': self.peak_amplitude,
            'rms_amplitude': self.rms_amplitude,
            'error_type': self.error_type,
            'error_message': self.error_message,
        }


# Immutable container aggregating pre-inference inspection metrics across an entire dataset
@dataclass(frozen=True)
class DatasetInspection(JsonResult):
    """
    Aggregates pre-inference validation results and Mel-frame statistics across a dataset
    of audio files or arrays without running GPU inference.

    Attributes:
        total_count (int): Total number of audio items inspected.
        eligible_count (int): Number of valid items with `1 <= actual_mel_frames <= 1500`.
        exceeds_1500_count (int): Number of decoded items rejected because `actual_mel_frames > 1500`.
        invalid_count (int): Number of items rejected due to missing/corrupt/empty audio.
        eligible_ratio (float): Proportion of inspected items that are eligible (`0.0` to `1.0`).
        total_duration_seconds (float): Sum of durations (in seconds) across all decoded items.
        eligible_duration_seconds (float): Sum of durations (in seconds) across all eligible items.
        min_mel_frames (int | None): Minimum Mel frame count among decoded items, or None if none decoded.
        mean_mel_frames (float | None): Arithmetic mean Mel frame count among decoded items, or None.
        max_mel_frames (int | None): Maximum Mel frame count among decoded items, or None if none decoded.
        items (list[AudioInspection]): Ordered per-item `AudioInspection` results.
    """
    total_count: int
    eligible_count: int
    exceeds_1500_count: int
    invalid_count: int
    eligible_ratio: float
    total_duration_seconds: float
    eligible_duration_seconds: float
    min_mel_frames: int | None
    mean_mel_frames: float | None
    max_mel_frames: int | None
    items: list[AudioInspection]

    # Return only the AudioInspection items that passed validation and satisfy <= 1500 Mel frames
    def eligible_items(self) -> list[AudioInspection]:
        """
        Filter and return only the valid (`status == 'eligible'`) audio inspection entries.

        Returns:
            list[AudioInspection]: List of eligible `AudioInspection` objects.
        """
        return [item for item in self.items if item.valid]

    # Return only the AudioInspection items that were rejected (>1500 frames or invalid audio)
    def rejected_items(self) -> list[AudioInspection]:
        """
        Filter and return only the rejected (`valid == False`) audio inspection entries.

        Returns:
            list[AudioInspection]: List of rejected `AudioInspection` objects.
        """
        return [item for item in self.items if not item.valid]

    # Return the filesystem paths of all eligible audio files for direct use in transcribe_many()
    def eligible_paths(self) -> list[str]:
        """
        Return the list of `source` file paths for all eligible (`valid == True`) file items,
        allowing callers to pass `dataset_info.eligible_paths()` directly to `model.transcribe_many()`.

        Returns:
            list[str]: Ordered list of eligible file path strings.
        """
        return [item.source for item in self.items if item.valid and item.source is not None]

    # Render a comprehensive human-readable summary report for the inspected dataset
    def to_report(self) -> str:
        """
        Build a multi-line human-readable dataset health report summarizing counts, durations,
        Mel-frame distribution, and any rejected items (`>1500` frames or corrupt audio).

        Variables:
            lines (list[str]): Accumulated report lines.
            rejected (list[AudioInspection]): Subset of items where `valid` is False.
            item (AudioInspection): Current rejected item being formatted.

        Returns:
            str: Multi-line English dataset inspection report.
        """
        lines: list[str]
        rejected: list[AudioInspection]
        item: AudioInspection

        lines = [
            '=== KorkutWhisper Dataset Inspection Report ===',
            f'Total Items         : {self.total_count}',
            f'Eligible (<=1500)   : {self.eligible_count} ({self.eligible_ratio * 100.0:.1f}%)',
            f'Exceeds 1500 Frames : {self.exceeds_1500_count}',
            f'Invalid / Corrupt   : {self.invalid_count}',
            f'Audio Duration      : eligible={self.eligible_duration_seconds:.2f}s / total={self.total_duration_seconds:.2f}s',
        ]
        if self.min_mel_frames is not None and self.mean_mel_frames is not None and self.max_mel_frames is not None:
            lines.append(
                f'Mel Frame Stats     : min={self.min_mel_frames} | '
                f'mean={self.mean_mel_frames:.1f} | max={self.max_mel_frames}'
            )

        rejected = self.rejected_items()
        if rejected:
            lines.append('Rejected Items      :')
            for item in rejected:
                lines.append(
                    f'  - [{item.status}] {item.audio_id} ({item.actual_mel_frames} frames, '
                    f'{item.duration:.2f}s): {item.error_message}'
                )
        return '\n'.join(lines)

    # Serialize all per-item inspections in the dataset to a CSV string
    def to_csv(self) -> str:
        """
        Format all `AudioInspection` items in this dataset into a CSV string with headers
        matching `INSPECTION_CSV_COLUMNS`.

        Variables:
            buf (io.StringIO): In-memory string buffer accumulating CSV rows.
            writer (csv.DictWriter): CSV writer configured with `INSPECTION_CSV_COLUMNS`.
            idx (int): Zero-based item index.
            item (AudioInspection): Current item being written.

        Returns:
            str: Complete CSV table string.
        """
        buf: io.StringIO
        writer: csv.DictWriter[str]
        idx: int
        item: AudioInspection

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(INSPECTION_CSV_COLUMNS), lineterminator='\n')
        writer.writeheader()
        for idx, item in enumerate(self.items):
            writer.writerow(item.to_csv_row(index=idx))
        return buf.getvalue()

    # Save the dataset inspection CSV table to a file on disk
    def save_csv(self, path: str | Path) -> Path:
        """
        Write the per-item dataset inspection table to a UTF-8 CSV file at `path`.

        Args:
            path (str | Path): Destination CSV file path.

        Variables:
            target (Path): Resolved destination `Path` object.

        Returns:
            Path: The destination CSV path written to disk.
        """
        target: Path
        target = Path(path)
        target.write_text(self.to_csv(), encoding='utf-8')
        return target



