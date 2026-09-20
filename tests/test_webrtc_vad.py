import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
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


@pytest.mark.parametrize("sr", [8000, 16000, 32000, 48000])
@pytest.mark.parametrize("frame_ms", [10, 20, 30])
def test_webrtc_vad_all_supported_sample_rates_and_durations(sr, frame_ms):
    """
    Day 4 Unit Test: Supported sample rates (8k, 16k, 32k, 48k) & valid frame durations (10, 20, 30 ms).
    DSP Formula: N = f_s * (frame_ms / 1000)
    Bytes: len = N * 2 (16-bit PCM)
    """
    vad = WebRTCVAD(mode=2, frame_duration_ms=frame_ms, sample_rate=sr)
    expected_samples = int(sr * (frame_ms / 1000.0))
    expected_bytes = expected_samples * 2

    assert vad.sample_rate == sr
    assert vad.frame_duration_ms == frame_ms
    assert vad.frame_samples == expected_samples
    assert vad.frame_bytes_len == expected_bytes

    # Should process silence frame of exact expected size without error
    silence_frame = b"\x00" * expected_bytes
    assert vad.is_speech(silence_frame) is False


def test_webrtc_vad_invalid_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        WebRTCVAD(mode=4)
    with pytest.raises(ValueError, match="Invalid mode"):
        WebRTCVAD(mode=-1)


def test_webrtc_vad_invalid_frame_duration():
    with pytest.raises(ValueError, match="Invalid frame duration"):
        WebRTCVAD(mode=2, frame_duration_ms=25)
    with pytest.raises(ValueError, match="Invalid frame duration"):
        WebRTCVAD(mode=2, frame_duration_ms=5)


def test_webrtc_vad_invalid_sample_rate():
    with pytest.raises(ValueError, match="Invalid sample rate"):
        WebRTCVAD(mode=2, sample_rate=44100)
    with pytest.raises(ValueError, match="Invalid sample rate"):
        WebRTCVAD(mode=2, sample_rate=22050)


# ---------------------------------------------------------------------------
# Frame Decision Tests: Valid & Invalid Frame Sizes, Silence
# ---------------------------------------------------------------------------

def test_is_speech_silence_frame():
    vad = WebRTCVAD(mode=2, frame_duration_ms=20, sample_rate=16000)
    # 640 bytes of silence (zero PCM)
    silence_frame = b"\x00" * 640
    assert vad.is_speech(silence_frame) is False


@pytest.mark.parametrize("bad_size", [1, 320, 639, 641, 1000])
def test_is_speech_invalid_frame_sizes_boundary(bad_size):
    """
    Day 4 Unit Test: Invalid frame length verification.
    WebRTC VAD strictly mandates exact frame byte lengths (e.g. 640 bytes for 20ms @ 16kHz).
    """
    vad = WebRTCVAD(mode=2, frame_duration_ms=20, sample_rate=16000)
    with pytest.raises(ValueError, match="Frame size mismatch"):
        vad.is_speech(b"\x00" * bad_size)


def test_process_frames():
    vad = WebRTCVAD(mode=2, frame_duration_ms=20, sample_rate=16000)
    frames = [b"\x00" * 640 for _ in range(10)]
    decisions = vad.process_frames(frames)
    assert len(decisions) == 10
    assert all(isinstance(d, bool) for d in decisions)
    assert all(d is False for d in decisions)  # all silence


# ---------------------------------------------------------------------------
# Silence & Empty Inputs Tests
# ---------------------------------------------------------------------------

def test_detect_speech_segments_pure_silence_numpy():
    """
    Day 4 Unit Test: Pure silence numpy array input.
    VAD should return empty segment list and 0.0 speech duration.
    """
    vad = WebRTCVAD(mode=2, frame_duration_ms=20, sample_rate=16000)
    silence_audio = np.zeros(32000, dtype=np.float32)  # 2.0s of silence
    segments, stats = vad.detect_speech_segments(silence_audio)

    assert segments == []
    assert stats["audio_duration_seconds"] == 2.0
    assert stats["speech_duration_retained_seconds"] == 0.0
    assert stats["speech_retained_ratio"] == 0.0
    assert stats["total_frames"] == 100
    assert stats["speech_frames"] == 0
    assert stats["final_segments_count"] == 0


def test_detect_speech_segments_pure_silence_wav(tmp_path):
    """
    Day 4 Unit Test: Pure silence WAV file input.
    """
    silence_wav = tmp_path / "silence_test.wav"
    sf.write(str(silence_wav), np.zeros(16000, dtype=np.float32), 16000)

    vad = WebRTCVAD(mode=2, frame_duration_ms=20, sample_rate=16000)
    segments, stats = vad.detect_speech_segments(silence_wav)

    assert segments == []
    assert stats["speech_duration_retained_seconds"] == 0.0
    assert stats["speech_retained_ratio"] == 0.0
    assert stats["speech_frames"] == 0


def test_detect_speech_segments_empty_inputs():
    """
    Day 4 Unit Test: Empty audio inputs (empty array / empty bytes).
    """
    vad = WebRTCVAD(mode=2, frame_duration_ms=20, sample_rate=16000)
    segments, stats = vad.detect_speech_segments(np.array([], dtype=np.float32))
    assert segments == []
    assert stats["total_frames"] == 0
    assert stats["audio_duration_seconds"] == 0.0

    segments_b, stats_b = vad.detect_speech_segments(b"")
    assert segments_b == []
    assert stats_b["total_frames"] == 0


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

