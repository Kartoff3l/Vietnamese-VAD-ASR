import sys
from pathlib import Path
import pytest

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import Segment
from src.segmenter import (
    frames_to_segments,
    merge_segments,
    pad_and_clamp_segments,
    filter_short_segments,
    postprocess_segments
)


# ---------------------------------------------------------------------------
# Segment Schema Tests
# ---------------------------------------------------------------------------

def test_segment_duration():
    seg = Segment(start_s=1.0, end_s=3.5, source="webrtc")
    assert seg.duration == 2.5


def test_segment_serialization():
    seg = Segment(start_s=0.5, end_s=1.2, source="webrtc_mode_2", confidence=0.95)
    d = seg.to_dict()
    assert d["start_s"] == 0.5
    assert d["end_s"] == 1.2
    assert d["duration"] == 0.7
    assert d["source"] == "webrtc_mode_2"
    assert d["confidence"] == 0.95

    reconstructed = Segment.from_dict(d)
    assert reconstructed.start_s == 0.5
    assert reconstructed.end_s == 1.2
    assert reconstructed.source == "webrtc_mode_2"


# ---------------------------------------------------------------------------
# frames_to_segments Tests
# ---------------------------------------------------------------------------

def test_frames_to_segments_empty():
    assert frames_to_segments([]) == []


def test_frames_to_segments_all_silence():
    assert frames_to_segments([False, False, False], frame_duration_ms=20) == []


def test_frames_to_segments_continuous():
    # 5 frames of speech (5 * 20ms = 100ms = 0.1s)
    decisions = [True, True, True, True, True]
    segments = frames_to_segments(decisions, frame_duration_ms=20)
    assert len(segments) == 1
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 0.10


def test_frames_to_segments_multiple_intervals():
    # 2 frames speech (0.00-0.04), 2 frames silence (0.04-0.08), 3 frames speech (0.08-0.14)
    decisions = [True, True, False, False, True, True, True]
    segments = frames_to_segments(decisions, frame_duration_ms=20)
    assert len(segments) == 2
    assert segments[0].start_s == 0.00
    assert segments[0].end_s == 0.04
    assert segments[1].start_s == 0.08
    assert segments[1].end_s == 0.14


# ---------------------------------------------------------------------------
# merge_segments Tests
# ---------------------------------------------------------------------------

def test_merge_segments_empty():
    assert merge_segments([]) == []


def test_merge_segments_close_gap():
    # Gap is 0.10s (100ms) <= merge_gap_ms 250ms -> should merge
    seg1 = Segment(start_s=0.0, end_s=1.0)
    seg2 = Segment(start_s=1.1, end_s=2.5)
    merged = merge_segments([seg1, seg2], merge_gap_ms=250)
    assert len(merged) == 1
    assert merged[0].start_s == 0.0
    assert merged[0].end_s == 2.5


def test_merge_segments_large_gap():
    # Gap is 0.50s (500ms) > merge_gap_ms 250ms -> should not merge
    seg1 = Segment(start_s=0.0, end_s=1.0)
    seg2 = Segment(start_s=1.5, end_s=2.5)
    merged = merge_segments([seg1, seg2], merge_gap_ms=250)
    assert len(merged) == 2
    assert merged[0].start_s == 0.0
    assert merged[0].end_s == 1.0
    assert merged[1].start_s == 1.5
    assert merged[1].end_s == 2.5


# ---------------------------------------------------------------------------
# pad_and_clamp_segments Tests
# ---------------------------------------------------------------------------

def test_pad_and_clamp_boundaries():
    # Segment [0.1, 1.0], pad_ms=200ms -> start should clamp to 0.0, end to 1.2
    seg = Segment(start_s=0.1, end_s=1.0)
    padded = pad_and_clamp_segments([seg], pad_ms=200, total_duration_s=2.0)
    assert len(padded) == 1
    assert padded[0].start_s == 0.0
    assert padded[0].end_s == 1.2


def test_pad_and_clamp_upper_limit():
    # Segment [1.0, 1.9], pad_ms=200ms, total_duration_s=2.0 -> end should clamp to 2.0
    seg = Segment(start_s=1.0, end_s=1.9)
    padded = pad_and_clamp_segments([seg], pad_ms=200, total_duration_s=2.0)
    assert len(padded) == 1
    assert padded[0].start_s == 0.8
    assert padded[0].end_s == 2.0


def test_pad_and_clamp_merges_overlap():
    # Two segments [0.5, 1.0] and [1.3, 2.0]
    # With pad 200ms: [0.3, 1.2] and [1.1, 2.2] -> they overlap at [1.1, 1.2] -> should merge to [0.3, 2.2]
    seg1 = Segment(start_s=0.5, end_s=1.0)
    seg2 = Segment(start_s=1.3, end_s=2.0)
    padded = pad_and_clamp_segments([seg1, seg2], pad_ms=200, total_duration_s=3.0)
    assert len(padded) == 1
    assert padded[0].start_s == 0.3
    assert padded[0].end_s == 2.2


# ---------------------------------------------------------------------------
# filter_short_segments Tests
# ---------------------------------------------------------------------------

def test_filter_short_segments():
    short_seg = Segment(start_s=0.0, end_s=0.15)  # 150ms < 300ms
    valid_seg = Segment(start_s=1.0, end_s=1.50)  # 500ms >= 300ms
    filtered = filter_short_segments([short_seg, valid_seg], min_segment_ms=300)
    assert len(filtered) == 1
    assert filtered[0].start_s == 1.0
    assert filtered[0].end_s == 1.50


# ---------------------------------------------------------------------------
# postprocess_segments Pipeline Tests
# ---------------------------------------------------------------------------

def test_postprocess_segments_end_to_end():
    # Raw segments:
    # 1. [0.2, 0.4] (200ms)
    # 2. [0.5, 0.7] (200ms, gap = 100ms -> merges with 1 to [0.2, 0.7])
    # 3. [2.0, 2.1] (100ms, isolated click -> after pad becomes 500ms, kept or filtered)
    raw = [
        Segment(start_s=0.2, end_s=0.4),
        Segment(start_s=0.5, end_s=0.7),
    ]
    processed = postprocess_segments(
        raw,
        pad_ms=200,
        merge_gap_ms=250,
        min_segment_ms=300,
        total_duration_s=5.0
    )
    assert len(processed) == 1
    # Merged: [0.2, 0.7] -> Padded: [0.0, 0.9]
    assert processed[0].start_s == 0.0
    assert processed[0].end_s == 0.9
