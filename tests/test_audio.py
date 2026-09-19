import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import inspect_audio, load_and_resample, float_to_pcm16, pcm16_to_float, frame_audio

SAMPLE_WAV = (
    PROJECT_ROOT
    / "tests" / "vivos" / "test" / "waves" / "VIVOSDEV01" / "VIVOSDEV01_R002.wav"
)


# ---------------------------------------------------------------------------
# inspect_audio Tests
# ---------------------------------------------------------------------------

def test_inspect_audio_returns_dict():
    result = inspect_audio(SAMPLE_WAV)
    assert isinstance(result, dict)
    expected_keys = {
        "file_name", "sample_rate", "channels",
        "duration_seconds", "total_samples", "subtype", "format"
    }
    assert expected_keys.issubset(result.keys())
    assert result["file_name"] == SAMPLE_WAV.name
    assert result["sample_rate"] > 0
    assert result["channels"] >= 1
    assert result["duration_seconds"] > 0


def test_inspect_audio_file_not_found():
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        inspect_audio("non_existent_audio_file_xyz.wav")


# ---------------------------------------------------------------------------
# load_and_resample Tests
# ---------------------------------------------------------------------------

def test_load_and_resample_output():
    audio, sr = load_and_resample(SAMPLE_WAV, target_sr=16000)
    assert sr == 16000
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1  # mono
    assert np.max(np.abs(audio)) <= 1.0


def test_load_and_resample_normalize_peak():
    audio, sr = load_and_resample(SAMPLE_WAV, target_sr=16000, normalize_peak=True)
    assert sr == 16000
    assert np.isclose(np.max(np.abs(audio)), 1.0, atol=1e-3)


def test_load_and_resample_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_and_resample("missing_file.wav")


# ---------------------------------------------------------------------------
# PCM16 Conversion Tests
# ---------------------------------------------------------------------------

def test_pcm16_float_roundtrip():
    original_float = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    pcm_bytes = float_to_pcm16(original_float)
    assert isinstance(pcm_bytes, bytes)
    assert len(pcm_bytes) == len(original_float) * 2  # 2 bytes per 16-bit sample

    recovered_float = pcm16_to_float(pcm_bytes)
    assert np.allclose(original_float, recovered_float, atol=1e-3)


def test_pcm16_empty():
    assert float_to_pcm16(np.array([], dtype=np.float32)) == b""
    assert len(pcm16_to_float(b"")) == 0


# ---------------------------------------------------------------------------
# frame_audio Tests
# ---------------------------------------------------------------------------

def test_frame_audio_numpy():
    # 1 second of audio at 16000 Hz = 16000 samples
    # 20 ms frames = 320 samples per frame -> 50 frames
    audio = np.zeros(16000, dtype=np.float32)
    frames = frame_audio(audio, frame_duration_ms=20, sample_rate=16000)
    assert len(frames) == 50
    assert all(len(f) == 320 for f in frames)


def test_frame_audio_bytes():
    # 1 second of audio at 16000 Hz = 16000 samples = 32000 bytes
    # 20 ms frame = 320 samples * 2 bytes = 640 bytes -> 50 frames
    pcm_bytes = b"\x00" * 32000
    frames = frame_audio(pcm_bytes, frame_duration_ms=20, sample_rate=16000)
    assert len(frames) == 50
    assert all(len(f) == 640 for f in frames)


def test_frame_audio_padding():
    # 350 samples: 1 full 320-sample frame + 1 padded 30-sample frame -> 2 frames
    audio = np.ones(350, dtype=np.float32)
    frames = frame_audio(audio, frame_duration_ms=20, sample_rate=16000, pad_last_frame=True)
    assert len(frames) == 2
    assert len(frames[0]) == 320
    assert len(frames[1]) == 320
    assert frames[1][30] == 0.0  # padded with zero


def test_frame_audio_invalid_type():
    with pytest.raises(TypeError):
        frame_audio([1, 2, 3])
