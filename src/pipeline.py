from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.asr import transcribe_full_audio

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration as a dictionary."""
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping")

    return config

def default_output_path(audio_path: str | Path) -> Path:
    """Build default JSON output path for full-audio baseline."""
    audio_name = Path(audio_path).stem
    return Path("results/baseline") / f"{audio_name}_full_audio.json"

def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Save result dictionary as readable UTF-8 JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok= True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def build_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run the Vietnamese ASR pipeline"
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to an input audio file.",
    )

    parser.add_argument(
        "--vad",
        default="none",
        choices=["none", "webrtc", "silero"],
        help="VAD method."
    )

    parser.add_argument(
        "--asr-model",
        default=None,
        help="Override ASR model from config, for example: base.",
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: results/baseline/<audio_name>_full_audio.json",
    )

    return parser


def print_summary(result: dict[str, Any], output_path: Path) -> None:
    """Print a concise command-line result summary"""
    print("\n=== Baseline Result ===")
    print(f"Pipeline: {result['pipeline']}")
    print(f"VAD method: {result['vad_method']}")
    print(f"Audio: {result['audio_path']}")
    print(f"Audio duration: {result['audio_duration_seconds']:.3f} s")
    print(f"ASR model: {result['asr_model']}")
    print(f"Language: {result['requested_language']}")
    print(f"ASR runtime: {result['asr_runtime_seconds']:.3f} s")
    print(f"ASR RTF: {result['asr_rtf']:.4f}")
    print(f"ASR segments: {result['num_asr_segments']}")
    print(f"Transcript: {result['transcript']}")
    print(f"Saved JSON: {output_path}")

def main() -> None:
    """Run whole-audio Vietnamese ASR baseline"""
    parser = build_parser()
    args = parser.parse_args()

    if args.vad != "none":
        parser.error(
            f"VAD mode '{args.vad}' is not implemented yet."
        )

    config = load_config(args.config)

    if "asr" not in config:
        raise ValueError("Missing 'asr' section in configuration file")

    asr_config = config["asr"]

    model_size = args.asr_model or asr_config["model"]
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(args.audio)
    )

    result = transcribe_full_audio(
        audio_path=args.audio,
        model_size=model_size,
        language=asr_config["language"],
        task=asr_config["task"],
        beam_size=asr_config["beam_size"],
        device=asr_config.get("device", "cpu"),
        compute_type=asr_config["compute_type"],
    )

    save_json(result, output_path)
    print_summary(result, output_path)

if __name__ == "__main__":
    main()