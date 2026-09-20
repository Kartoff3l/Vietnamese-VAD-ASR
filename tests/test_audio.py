import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
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
# load_and_resample Tests: Sample Rate, Stereo-to-Mono, Silence
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


@pytest.mark.parametrize("source_sr", [8000, 22050, 44100, 48000])
def test_load_and_resample_various_sample_rates(tmp_path, source_sr):
    """
    Day 4 Unit Test: Sample rate conversion & Nyquist preservation.
    Ensures polyphase resampling converts diverse input sample rates to 16,000 Hz.
    """
    duration_s = 1.5
    num_samples = int(source_sr * duration_s)
    # Generate 440 Hz test sine tone
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    synthetic_signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    temp_wav = tmp_path / f"test_sr_{source_sr}.wav"
    sf.write(str(temp_wav), synthetic_signal, source_sr)

    audio_resampled, target_sr = load_and_resample(temp_wav, target_sr=16000)

    assert target_sr == 16000
    assert audio_resampled.dtype == np.float32
    assert audio_resampled.ndim == 1
    # Expected output sample count should be ~16000 * 1.5 = 24000 samples (+- 1 sample due to rounding)
    expected_samples = int(16000 * duration_s)
    assert abs(len(audio_resampled) - expected_samples) <= 2


def test_load_and_resample_stereo_to_mono(tmp_path):
    """
    Day 4 Unit Test: Stereo-to-mono downmixing.
    Ensures multi-channel audio is averaged across channels to single-channel mono: (L + R) / 2.
    """
    duration_s = 1.0
    sr = 16000
    num_samples = int(sr * duration_s)

    # Channel 0 = 0.8 constant, Channel 1 = 0.2 constant -> Average should be 0.5
    channel_l = np.full(num_samples, 0.8, dtype=np.float32)
    channel_r = np.full(num_samples, 0.2, dtype=np.float32)
    stereo_audio = np.column_stack([channel_l, channel_r])

    temp_stereo_wav = tmp_path / "test_stereo.wav"
    sf.write(str(temp_stereo_wav), stereo_audio, sr)

    mono_audio, out_sr = load_and_resample(temp_stereo_wav, target_sr=16000)

    assert out_sr == 16000
    assert mono_audio.ndim == 1
    assert len(mono_audio) == num_samples
    assert np.allclose(mono_audio, 0.5, atol=1e-3)


def test_load_and_resample_multichannel_quad(tmp_path):
    """
    Day 4 Unit Test: 4-channel surround downmixing to 1D mono.
    """
    sr = 16000
    num_samples = 1600
    quad_audio = np.ones((num_samples, 4), dtype=np.float32) * 0.4

    temp_quad_wav = tmp_path / "test_quad.wav"
    sf.write(str(temp_quad_wav), quad_audio, sr)

    mono_audio, out_sr = load_and_resample(temp_quad_wav, target_sr=16000)
    assert out_sr == 16000
    assert mono_audio.ndim == 1
    assert np.allclose(mono_audio, 0.4, atol=1e-3)


def test_load_and_resample_pure_silence(tmp_path):
    """
    Day 4 Unit Test: Pure silence audio input.
    Ensures all-zero input does not cause division-by-zero even with normalize_peak=True.
    """
    sr = 16000
    silence = np.zeros(32000, dtype=np.float32)

    temp_silence_wav = tmp_path / "test_silence.wav"
    sf.write(str(temp_silence_wav), silence, sr)

    audio_resampled, out_sr = load_and_resample(temp_silence_wav, target_sr=16000, normalize_peak=True)
    assert out_sr == 16000
    assert audio_resampled.ndim == 1
    assert len(audio_resampled) == 32000
    assert np.all(audio_resampled == 0.0)


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


def test_pcm16_clipping_guard():
    """Ensures out-of-range floats are safely clipped to [-1.0, 1.0] before PCM16 conversion."""
    out_of_bounds = np.array([-2.5, -1.0, 0.0, 1.0, 3.5], dtype=np.float32)
    pcm_bytes = float_to_pcm16(out_of_bounds)
    recovered = pcm16_to_float(pcm_bytes)
    assert np.isclose(recovered[0], -1.0, atol=1e-3)
    assert np.isclose(recovered[-1], 1.0, atol=1e-3)


def test_pcm16_empty():
    assert float_to_pcm16(np.array([], dtype=np.float32)) == b""
    assert len(pcm16_to_float(b"")) == 0


# ---------------------------------------------------------------------------
# frame_audio Tests: Valid Frame Lengths & Padding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frame_ms,expected_samples", [(10, 160), (20, 320), (30, 480)])
def test_frame_audio_valid_frame_durations(frame_ms, expected_samples):
    """
    Day 4 Unit Test: Valid frame lengths (10ms, 20ms, 30ms) at 16,000 Hz.
    DSP Formula: N = f_s * T
    For 16000 samples (1.0s):
    - 10ms -> 160 samples -> 100 frames
    - 20ms -> 320 samples -> 50 frames
    - 30ms -> 480 samples -> 33 full frames + 1 zero-padded frame (160 samples + 320 padding) = 34 frames
    """
    import math
    audio = np.zeros(16000, dtype=np.float32)  # 1 second
    frames = frame_audio(audio, frame_duration_ms=frame_ms, sample_rate=16000, pad_last_frame=True)
    assert all(len(f) == expected_samples for f in frames)
    expected_total_frames = math.ceil(16000 / expected_samples)
    assert len(frames) == expected_total_frames

    # Unpadded should have floor(16000 / expected_samples) frames
    unpadded_frames = frame_audio(audio, frame_duration_ms=frame_ms, sample_rate=16000, pad_last_frame=False)
    assert len(unpadded_frames) == (16000 // expected_samples)



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


def test_frame_audio_unpadded_drops_remainder():
    """
    Day 4 Unit Test: Uneven audio framing without padding.
    Ensures remainder frames smaller than frame_samples are discarded when pad_last_frame=False.
    """
    audio = np.ones(350, dtype=np.float32)
    frames = frame_audio(audio, frame_duration_ms=20, sample_rate=16000, pad_last_frame=False)
    assert len(frames) == 1
    assert len(frames[0]) == 320


def test_frame_audio_invalid_type():
    with pytest.raises(TypeError):
        frame_audio([1, 2, 3])

