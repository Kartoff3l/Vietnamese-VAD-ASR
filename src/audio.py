from pathlib import Path
from typing import Dict, Any, Tuple, List, Union
import math
import numpy as np
import soundfile as sf
import scipy.signal


def inspect_audio(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspects an audio file and returns metadata dictionary."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    info = sf.info(str(file_path))

    return {
        "file_name": file_path.name,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_seconds": round(float(info.duration), 4),
        "total_samples": int(info.frames),
        "subtype": info.subtype,
        "format": info.format
    }


def load_and_resample(
    file_path: Union[str, Path],
    target_sr: int = 16000,
    normalize_peak: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file, converts it to mono float32, and resamples to target_sr.
    Returns (audio_np, target_sr).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio_data, orig_sr = sf.read(str(file_path), dtype="float32")

    # Convert to mono if multi-channel
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if sample rate doesn't match target
    if orig_sr != target_sr:
        # Calculate rational resampling factors
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        audio_data = scipy.signal.resample_poly(audio_data, up, down).astype(np.float32)

    if normalize_peak:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

    audio_data = np.clip(audio_data, -1.0, 1.0).astype(np.float32)
    return audio_data, target_sr


def float_to_pcm16(audio_float: np.ndarray) -> bytes:
    """Converts a float32 audio numpy array [-1.0, 1.0] to 16-bit signed PCM bytes."""
    clipped = np.clip(audio_float, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).round().astype(np.int16)
    return pcm16.tobytes()


def pcm16_to_float(pcm_bytes: bytes) -> np.ndarray:
    """Converts 16-bit signed PCM bytes to a float32 numpy array [-1.0, 1.0]."""
    if len(pcm_bytes) == 0:
        return np.array([], dtype=np.float32)
    pcm16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return (pcm16.astype(np.float32) / 32767.0)


def frame_audio(
    audio: Union[np.ndarray, bytes],
    frame_duration_ms: int = 20,
    sample_rate: int = 16000,
    pad_last_frame: bool = True
) -> List[Union[np.ndarray, bytes]]:
    """
    Splits audio into frames of duration frame_duration_ms.
    At 16000 Hz and 20 ms, each frame contains 320 samples (640 bytes for PCM16).
    """
    frame_samples = int(sample_rate * (frame_duration_ms / 1000.0))

    if isinstance(audio, (bytes, bytearray)):
        frame_bytes_len = frame_samples * 2  # 2 bytes per sample for 16-bit PCM
        total_bytes = len(audio)
        frames = []

        for offset in range(0, total_bytes, frame_bytes_len):
            chunk = audio[offset:offset + frame_bytes_len]
            if len(chunk) == frame_bytes_len:
                frames.append(chunk)
            elif pad_last_frame and len(chunk) > 0:
                padded_chunk = chunk + b"\x00" * (frame_bytes_len - len(chunk))
                frames.append(padded_chunk)
        return frames

    elif isinstance(audio, np.ndarray):
        total_samples = len(audio)
        frames = []

        for offset in range(0, total_samples, frame_samples):
            chunk = audio[offset:offset + frame_samples]
            if len(chunk) == frame_samples:
                frames.append(chunk)
            elif pad_last_frame and len(chunk) > 0:
                pad_width = frame_samples - len(chunk)
                padded_chunk = np.pad(chunk, (0, pad_width), mode="constant", constant_values=0)
                frames.append(padded_chunk)
        return frames
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}. Only numpy ndarray or bytes are supported.")