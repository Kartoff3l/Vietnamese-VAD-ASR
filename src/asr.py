from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import soundfile as sf
from faster_whisper import WhisperModel



def get_audio_duration(audio_path: str | Path) -> float:
    """Return audio duration in seconds."""
    return float(sf.info(str(audio_path)).duration)


def load_model(
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
) -> WhisperModel:
    """Load and return a faster-whisper model"""
    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

def transcribe_full_audio(
        audio_path: str | Path,
        model_size: str = "base",
        language: str = "vi",
        task: str = "transcribe",
        beam_size: int = 5,
        device: str = "cpu",
        compute_type: str = "int8",
) -> dict[str, Any]:
    """Transcribe a complete audio file without external VAD"""
    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio_duration_second = get_audio_duration(path)
    if audio_duration_second <= 0:
        raise ValueError(f"Audio duration must be positive: {path}")

    model = load_model(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )

    start_time = perf_counter()

    segments, info = model.transcribe(
        str(path),
        language=language,
        task=task,
        beam_size=beam_size,
        vad_filter=False,
    )

    segment_results: list[dict[str, Any]] = []
    transcript_parts: list[str] = []

    for segment in segments:
        text = segment.text.strip()

        segment_results.append(
            {
                "start_s": round(float (segment.start), 3),
                "end_s": round(float(segment.end),3),
                "text": text,
            }
        )

        if text:
            transcript_parts.append(text)

    asr_runtime_s = perf_counter() - start_time
    transcript = " ".join(transcript_parts)

    return{
        "audio_path": str(path),
        "pipeline": "full_audio",
        "vad_method": "none",
        "asr_model": model_size,
        "device": device,
        "compute_type": compute_type,
        "requested_language": language,
        "detected_language": info.language,
        "detected_language_probability": round(float(info.language_probability), 4,),
        "task": task,
        "beam_size": beam_size,
        "audio_duration_seconds": round(audio_duration_second, 3),
        "asr_runtime_seconds": round(asr_runtime_s, 3),
        "asr_rtf": round(asr_runtime_s / audio_duration_second, 4),
        "num_asr_segments": len(segment_results),
        "segments": segment_results,
        "transcript": transcript,
    }
