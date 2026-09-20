<<<<<<< HEAD
import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import Segment
from src.evaluation import segments_to_binary_mask, compute_vad_metrics, load_boundary_labels


def test_segments_to_binary_mask():
    # 2.0s duration with 10ms resolution = 200 steps
    # Segment [0.1, 0.3] -> steps 10 to 30 = True
    seg = Segment(start_s=0.1, end_s=0.3)
    mask = segments_to_binary_mask([seg], total_duration_s=2.0, resolution_ms=10)
    assert len(mask) == 200
    assert mask[5] is np.bool_(False)
    assert mask[15] is np.bool_(True)
    assert mask[25] is np.bool_(True)
    assert mask[35] is np.bool_(False)


def test_compute_vad_metrics_perfect_match():
    ref = [Segment(start_s=1.0, end_s=3.0)]
    pred = [Segment(start_s=1.0, end_s=3.0)]
    metrics = compute_vad_metrics(pred, ref, total_duration_s=5.0)

    assert metrics["tp_seconds"] == pytest.approx(2.0, abs=0.02)
    assert metrics["fp_seconds"] == pytest.approx(0.0, abs=0.02)
    assert metrics["fn_seconds"] == pytest.approx(0.0, abs=0.02)
    assert metrics["iou"] == pytest.approx(1.0, abs=0.02)
    assert metrics["precision"] == pytest.approx(1.0, abs=0.02)
    assert metrics["recall"] == pytest.approx(1.0, abs=0.02)


def test_compute_vad_metrics_false_alarm_and_miss():
    # Ref is [1.0, 2.0] (1.0s)
    # Pred is [0.5, 1.5] (1.0s)
    # TP: [1.0, 1.5] = 0.5s
    # FP: [0.5, 1.0] = 0.5s
    # FN: [1.5, 2.0] = 0.5s
    ref = [Segment(start_s=1.0, end_s=2.0)]
    pred = [Segment(start_s=0.5, end_s=1.5)]
    metrics = compute_vad_metrics(pred, ref, total_duration_s=5.0)

    assert metrics["tp_seconds"] == pytest.approx(0.5, abs=0.02)
    assert metrics["fp_seconds"] == pytest.approx(0.5, abs=0.02)
    assert metrics["fn_seconds"] == pytest.approx(0.5, abs=0.02)
    assert metrics["iou"] == pytest.approx(0.5 / 1.5, abs=0.02)  # 0.333
    assert metrics["precision"] == pytest.approx(0.5, abs=0.02)
    assert metrics["recall"] == pytest.approx(0.5, abs=0.02)


def test_load_boundary_labels():
    sample_json = PROJECT_ROOT / "data" / "boundaries" / "bud500_13.json"
    if sample_json.exists():
        segments = load_boundary_labels(sample_json)
        assert len(segments) == 1
        assert segments[0].start_s == 0.07
        assert segments[0].end_s == 1.69
=======
import pytest

from src.evaluation import(
    calculate_rtf,
    calculate_text_metrics,
    normalize_vietnamese_text,
)

def test_normalize_vietnamese_text():
    text= " Xin chào, tôi đến từ Biên Hòa"

    assert(
        normalize_vietnamese_text(text) == "xin chào tôi đến từ biên hòa"
    )

def test_normalize_preserves_vietnamese_diacritics():
    assert(
        normalize_vietnamese_text("dấu câu Tiếng Việt") == "dấu câu tiếng việt"
    )

def test_identical_text_has_zero_error():
    result = calculate_text_metrics(
        reference= "Hôm nay bạn có đi Bình Dương không?",
        prediction= "hôm nay bạn có đi bình dương không",
    )
    assert result["wer"] == 0.0
    assert result["cer"] == 0.0

def test_rtf():
    assert calculate_rtf(2.5, 10.0) == 0.25

def test_rtf_rejects_zero_duration():
    with pytest.raises(ValueError):
        calculate_rtf(2.0, 0.0)

def test_empty_reference_is_rejected():
    with pytest.raises(ValueError):
        calculate_text_metrics(
            reference= " ",
            prediction="xin chào",
        )
>>>>>>> feature/asr-evaluation-api
