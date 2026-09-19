import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.webrtc_vad import WebRTCVAD, run_webrtc_vad
from src.schemas import Segment

SAMPLE_WAV = (
    PROJECT_ROOT
    / "tests" / "vivos" / "test" / "waves" / "VIVOSDEV01" / "VIVOSDEV01_R002.wav"
)


# ---------------------------------------------------------------------------
# Initialization & Validation Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_webrtc_vad_valid_modes(mode):
    vad = WebRTCVAD(mode=mode, frame_duration_ms=20, sample_rate=16000)
    assert vad.mode == mode
    assert vad.frame_samples == 320
    assert vad.frame_bytes_len == 640


def test_webrtc_vad_invalid_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        WebRTCVAD(mode=4)
    with pytest.raises(ValueError, match="Invalid mode"):
        WebRTCVAD(mode=-1)


def test_webrtc_vad_invalid_frame_duration():
    with pytest.raises(ValueError, match="Invalid frame duration"):
        WebRTCVAD(mode=2, frame_duration_ms=25)


def test_webrtc_vad_invalid_sample_rate():
    with pytest.raises(ValueError, match="Invalid sample rate"):
        WebRTCVAD(mode=2, sample_rate=44100)


# ---------------------------------------------------------------------------
# Frame Decision Tests
# ---------------------------------------------------------------------------

def test_is_speech_silence_frame():
    vad = WebRTCVAD(mode=2, frame_duration_ms=20, sample_rate=16000)
    # 640 bytes of silence (zero PCM)
    silence_frame = b"\x00" * 640
    assert vad.is_speech(silence_frame) is False


def test_is_speech_frame_size_mismatch():
    vad = WebRTCVAD(mode=2, frame_duration_ms=20, sample_rate=16000)
    # Wrong size (320 bytes instead of 640 bytes)
    with pytest.raises(ValueError, match="Frame size mismatch"):
        vad.is_speech(b"\x00" * 320)


def test_process_frames():
    vad = WebRTCVAD(mode=2, frame_duration_ms=20, sample_rate=16000)
    frames = [b"\x00" * 640 for _ in range(10)]
    decisions = vad.process_frames(frames)
    assert len(decisions) == 10
    assert all(isinstance(d, bool) for d in decisions)
    assert all(d is False for d in decisions)  # all silence


# ---------------------------------------------------------------------------
# End-to-End Segmentation on Real Audio
# ---------------------------------------------------------------------------

def test_detect_speech_segments_real_audio():
    vad = WebRTCVAD(mode=2, frame_duration_ms=20, sample_rate=16000)
    segments, stats = vad.detect_speech_segments(
        SAMPLE_WAV,
        pad_ms=200,
        merge_gap_ms=250,
        min_segment_ms=300
    )

    assert isinstance(segments, list)
    assert len(segments) >= 1
    assert all(isinstance(seg, Segment) for seg in segments)

    # Check stats dictionary
    expected_stats = {
        "audio_duration_seconds", "speech_duration_retained_seconds",
        "speech_retained_ratio", "total_frames", "speech_frames",
        "raw_segments_count", "final_segments_count", "aggressiveness_mode"
    }
    assert expected_stats.issubset(stats.keys())
    assert stats["audio_duration_seconds"] > 0
    assert stats["speech_duration_retained_seconds"] <= stats["audio_duration_seconds"]
    assert 0.0 < stats["speech_retained_ratio"] <= 1.0
    assert stats["aggressiveness_mode"] == 2


def test_aggressiveness_modes_comparison():
    # Higher aggressiveness mode (Mode 3) should filter more or equal speech frames than Mode 1
    vad_mode1 = WebRTCVAD(mode=1, frame_duration_ms=20, sample_rate=16000)
    vad_mode3 = WebRTCVAD(mode=3, frame_duration_ms=20, sample_rate=16000)

    _, stats1 = vad_mode1.detect_speech_segments(SAMPLE_WAV)
    _, stats3 = vad_mode3.detect_speech_segments(SAMPLE_WAV)

    assert stats1["speech_frames"] >= stats3["speech_frames"]
    assert stats1["speech_retained_ratio"] >= stats3["speech_retained_ratio"]


def test_run_webrtc_vad_convenience_function():
    segments, stats = run_webrtc_vad(SAMPLE_WAV, mode=2)
    assert len(segments) >= 1
    assert stats["aggressiveness_mode"] == 2
