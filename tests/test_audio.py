"""
Tests for src/audio.py — dùng file .wav thật từ dataset VIVOS.
"""

import sys
import pytest
from pathlib import Path

# Đảm bảo import được module từ src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio import inspect_audio

# Đường dẫn tới một file .wav thật trong dataset
SAMPLE_WAV = (
    Path(__file__).parent
    / "vivos" / "test" / "waves" / "VIVOSDEV01" / "VIVOSDEV01_R002.wav"
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_returns_dict():
    """Hàm phải trả về một dict."""
    result = inspect_audio(SAMPLE_WAV)
    assert isinstance(result, dict)


def test_required_keys():
    """Dict kết quả phải có đủ 7 trường."""
    result = inspect_audio(SAMPLE_WAV)
    expected_keys = {"file_name", "sample_rate", "channels",
                     "duration_seconds", "total_samples", "subtype", "format"}
    assert expected_keys.issubset(result.keys())


def test_file_name():
    """file_name phải khớp với tên file trên đĩa."""
    result = inspect_audio(SAMPLE_WAV)
    assert result["file_name"] == SAMPLE_WAV.name


def test_sample_rate_positive():
    """sample_rate phải là số nguyên dương."""
    result = inspect_audio(SAMPLE_WAV)
    assert isinstance(result["sample_rate"], int)
    assert result["sample_rate"] > 0


def test_channels_positive():
    """channels phải là số nguyên dương (thường là 1 hoặc 2)."""
    result = inspect_audio(SAMPLE_WAV)
    assert isinstance(result["channels"], int)
    assert result["channels"] >= 1


def test_duration_positive():
    """duration_seconds phải là số dương."""
    result = inspect_audio(SAMPLE_WAV)
    assert result["duration_seconds"] > 0


def test_duration_rounded():
    """duration_seconds phải được làm tròn 4 chữ số thập phân."""
    result = inspect_audio(SAMPLE_WAV)
    duration = result["duration_seconds"]
    assert round(duration, 4) == duration


def test_total_samples_positive():
    """total_samples phải là số nguyên dương."""
    result = inspect_audio(SAMPLE_WAV)
    assert result["total_samples"] > 0


def test_accepts_string_path():
    """Hàm phải chấp nhận đường dẫn dạng str (không chỉ Path)."""
    result = inspect_audio(str(SAMPLE_WAV))
    assert result["file_name"] == SAMPLE_WAV.name


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

def test_file_not_found():
    """Phải raise FileNotFoundError khi file không tồn tại."""
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        inspect_audio("non_existent_file.wav")
