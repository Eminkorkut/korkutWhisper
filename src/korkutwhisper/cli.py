"""Command-line interface (CLI) for KorkutWhisper transcription, inspection, and profile verification."""

from __future__ import annotations
import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any
from .api import KorkutWhisper
from .audio import inspect_audio, inspect_dataset
from .config import GateConfig, InferenceConfig, KorkutConfig, RuntimeConfig, WhisperConfig
from .exceptions import KorkutWhisperError, ConfigurationError
from .profiles import load_gate_profile
from .types import format_batch_csv, save_batch_csv, save_batch_jsonl


# Build the argument parser for the korkutwhisper command-line interface
def build_parser() -> argparse.ArgumentParser:
    """
    Construct and configure the `argparse.ArgumentParser` supporting single-file and batch
    transcription, CPU-only audio/dataset inspection (`--inspect`), model metadata inspection
    (`--model-info`), and G2 gate profile verification (`--gate-info`).

    Variables:
        parser (argparse.ArgumentParser): Configured top-level CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser ready to parse command-line arguments.
    """
    parser: argparse.ArgumentParser
    parser = argparse.ArgumentParser(
        prog='korkutwhisper',
        description='Decoder-guided selective CTranslate2 Whisper inference and audio inspection CLI.',
    )

    # Input targets (one or more audio files or directories)
    parser.add_argument(
        'inputs',
        nargs='*',
        help='One or more audio file paths (or directory paths when using --inspect).',
    )

    # Model and configuration options
    parser.add_argument(
        '-m',
        '--model',
        dest='model_path',
        default=None,
        help='Path to the local CTranslate2 Whisper model directory.',
    )
    parser.add_argument(
        '-c',
        '--config',
        dest='config_path',
        default=None,
        help='Path to a JSON KorkutConfig file (overrides individual model flags).',
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=('cuda', 'cpu', 'auto'),
        help="Execution device ('cuda', 'cpu', or 'auto'; default: 'cuda').",
    )
    parser.add_argument(
        '--compute-type',
        default='int8',
        help="CTranslate2 quantization/compute type (default: 'int8').",
    )
    parser.add_argument(
        '--device-index',
        type=int,
        default=0,
        help='GPU device index when running on CUDA (default: 0).',
    )
    parser.add_argument(
        '--cpu-threads',
        type=int,
        default=0,
        help='Number of CPU threads (0 for automatic; default: 0).',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of CTranslate2 worker threads (default: 1).',
    )
    parser.add_argument(
        '--beam-size',
        type=int,
        default=5,
        help='Beam search decoding width (default: 5).',
    )
    parser.add_argument(
        '--language',
        default='tr',
        help="Target spoken language code (default: 'tr').",
    )
    parser.add_argument(
        '--task',
        default='transcribe',
        choices=('transcribe', 'translate'),
        help="Whisper decoding task ('transcribe' or 'translate'; default: 'transcribe').",
    )

    # Strategy, seeding, gate profile, and warm-up options
    parser.add_argument(
        '--strategy',
        default='g2',
        choices=('g2', 'fixed_1500', 'fixed_3000'),
        help="Encoder frame routing strategy (default: 'g2').",
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base integer seed for deterministic CTranslate2 RNG seeding (default: 42).',
    )
    parser.add_argument(
        '--profile',
        default='decoder_based_b2',
        help="Built-in G2 gate profile identifier (default: 'decoder_based_b2').",
    )
    parser.add_argument(
        '--profile-path',
        default=None,
        help='Optional path to an external custom G2 gate profile JSON file.',
    )
    parser.add_argument(
        '--allow-unverified',
        action='store_true',
        help='Allow running G2 on unverified models, runtime versions, or custom gate profiles.',
    )
    parser.add_argument(
        '--warmup',
        action='store_true',
        help='Run a synthetic 1-second silent warm-up pass before transcribing real audio.',
    )

    # Output formatting and file saving options
    parser.add_argument(
        '-f',
        '--format',
        dest='output_format',
        default='text',
        choices=('text', 'txt', 'report', 'json', 'csv', 'srt', 'vtt'),
        help="Output format for console display or file export (default: 'text').",
    )
    parser.add_argument(
        '-o',
        '--output',
        dest='output_path',
        default=None,
        help='Optional file path to save the output (supports .txt, .json, .csv, .srt, .vtt, .jsonl).',
    )

    # Standalone inspection and diagnostic modes
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Inspect audio file(s) or dataset directory on CPU without loading or running the GPU model.',
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively scan subdirectories when --inspect is given a directory.',
    )
    parser.add_argument(
        '--pattern',
        default=None,
        help="Optional glob pattern when scanning a directory with --inspect (e.g. '*.wav').",
    )
    parser.add_argument(
        '--gate-info',
        action='store_true',
        help='Print the G2 gate profile parameters and SHA-256 verification status as JSON and exit.',
    )
    parser.add_argument(
        '--model-info',
        action='store_true',
        help='Load the model, print its ModelInfo metadata as JSON, and exit.',
    )
    return parser


# Execute the CLI workflow from parsed command-line arguments
def main(argv: Sequence[str] | None = None) -> int:
    """
    Parse command-line arguments and execute the requested operation (`--gate-info`,
    `--inspect`, `--model-info`, single-file transcription, or batch transcription).

    Args:
        argv (Sequence[str] | None): Optional argument list for programmatic/test invocation.
            Defaults to `sys.argv[1:]` when None.

    Variables:
        parser (argparse.ArgumentParser): CLI argument parser built by `build_parser()`.
        args (argparse.Namespace): Parsed command-line options.
        profile_obj (Any): Loaded `GateProfile` when `--gate-info` is requested.
        config (KorkutConfig): Constructed or loaded composite configuration for transcription.
        out_path (Path): Destination file path when `--output` is provided.

    Returns:
        int: Process exit code (`0` on success, `1` on handled `KorkutWhisperError`).
    """
    parser: argparse.ArgumentParser
    args: argparse.Namespace

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        # Mode 1: Print G2 gate profile parameters and SHA-256 verification status (no GPU/model required)
        if args.gate_info:
            profile_obj = load_gate_profile(args.profile, args.profile_path)
            import json
            print(json.dumps(profile_obj.to_dict(), ensure_ascii=False, indent=2))
            return 0

        # Mode 2: CPU-only pre-inference audio / dataset inspection (no GPU/model required)
        if args.inspect:
            if not args.inputs:
                raise ConfigurationError('--inspect requires at least one audio file or directory path')

            if len(args.inputs) == 1 and Path(args.inputs[0]).is_file():
                single_info = inspect_audio(args.inputs[0])
                if args.output_format == 'json':
                    text_out = single_info.to_json(indent=2)
                elif args.output_format == 'csv':
                    ds_wrap = inspect_dataset([args.inputs[0]])
                    text_out = ds_wrap.to_csv().rstrip('\n')
                else:
                    text_out = single_info.to_report()

                if args.output_path:
                    Path(args.output_path).write_text(text_out + '\n', encoding='utf-8')
                print(text_out)
                return 0 if single_info.valid else 1

            ds_target: Any = args.inputs[0] if len(args.inputs) == 1 else args.inputs
            ds_info = inspect_dataset(ds_target, pattern=args.pattern, recursive=args.recursive)
            if args.output_format == 'json':
                text_out = ds_info.to_json(indent=2)
            elif args.output_format == 'csv':
                text_out = ds_info.to_csv().rstrip('\n')
            else:
                text_out = ds_info.to_report()

            if args.output_path:
                out_path = Path(args.output_path)
                if out_path.suffix.lower() == '.csv':
                    ds_info.save_csv(out_path)
                elif out_path.suffix.lower() == '.json':
                    ds_info.save_json(out_path)
                else:
                    out_path.write_text(text_out + '\n', encoding='utf-8')
            print(text_out)
            return 0 if ds_info.exceeds_1500_count == 0 and ds_info.invalid_count == 0 else 1

        # Build or load KorkutConfig for model-backed operations (--model-info or transcription)
        if args.config_path is not None:
            config = KorkutConfig.load_json(args.config_path)
        else:
            if not args.model_path:
                raise ConfigurationError('--model (or --config) is required for transcription')
            config = KorkutConfig(
                whisper=WhisperConfig(
                    model_path=args.model_path,
                    device=args.device,
                    compute_type=args.compute_type,
                    device_index=args.device_index,
                    cpu_threads=args.cpu_threads,
                    num_workers=args.num_workers,
                    beam_size=args.beam_size,
                    language=args.language,
                    task=args.task,
                ),
                inference=InferenceConfig(
                    strategy=args.strategy,
                    seed=args.seed,
                ),
                gate=GateConfig(
                    profile=args.profile,
                    allow_unverified=args.allow_unverified,
                    profile_path=args.profile_path,
                ),
                runtime=RuntimeConfig(
                    warmup=args.warmup,
                ),
            )

        with KorkutWhisper(config=config) as model:
            # Mode 3: Print loaded ModelInfo JSON and exit
            if args.model_info:
                print(model.get_model_info().to_json(indent=2))
                return 0

            if not args.inputs:
                raise ConfigurationError('at least one input audio file path is required')

            # Mode 4: Single audio file transcription
            if len(args.inputs) == 1:
                result = model.transcribe(args.inputs[0])
                formatted = result.format(args.output_format)
                if args.output_path is not None:
                    result.save(args.output_path, fmt=args.output_format if args.output_format != 'text' else None)
                print(formatted.rstrip('\n'))
                return 0

            # Mode 5: Sequential batch transcription for multiple audio files
            items = model.transcribe_many(args.inputs)
            if args.output_path is not None:
                out_path = Path(args.output_path)
                if out_path.suffix.lower() == '.jsonl':
                    save_batch_jsonl(items, out_path)
                else:
                    save_batch_csv(items, out_path)

            if args.output_format == 'csv':
                print(format_batch_csv(items).rstrip('\n'))
            elif args.output_format == 'json':
                import json
                print(json.dumps([item.to_dict() for item in items], ensure_ascii=False, indent=2))
            else:
                for item in items:
                    if item.result is not None:
                        if args.output_format == 'report':
                            print(item.result.to_report())
                        else:
                            print(f'[{item.index}] {item.result.text}')
                    else:
                        print(f'[{item.index}] ERROR ({item.error_type}): {item.error_message}')
            return 0 if all(item.result is not None for item in items) else 1

    except KorkutWhisperError as exc:
        print(f'Error ({type(exc).__name__}): {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
