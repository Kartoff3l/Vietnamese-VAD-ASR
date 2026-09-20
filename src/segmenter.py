from typing import List, Optional
from src.schemas import Segment


def frames_to_segments(
    frame_decisions: List[bool],
    frame_duration_ms: int = 20,
    source: str = "webrtc"
) -> List[Segment]:
    """
    Converts a sequence of boolean frame-level speech decisions into timestamped Segments.
    
    Args:
        frame_decisions: List of booleans where True indicates speech.
        frame_duration_ms: Duration of each frame in milliseconds (default 20ms).
        source: Source identifier string (default 'webrtc').
        
    Returns:
        List of raw Segment objects.
    """
    if not frame_decisions:
        return []

    frame_duration_s = frame_duration_ms / 1000.0
    segments: List[Segment] = []
    in_speech = False
    start_frame = 0

    for idx, is_speech in enumerate(frame_decisions):
        if is_speech and not in_speech:
            in_speech = True
            start_frame = idx
        elif not is_speech and in_speech:
            in_speech = False
            start_s = round(start_frame * frame_duration_s, 4)
            end_s = round(idx * frame_duration_s, 4)
            segments.append(Segment(start_s=start_s, end_s=end_s, source=source))

    if in_speech:
        start_s = round(start_frame * frame_duration_s, 4)
        end_s = round(len(frame_decisions) * frame_duration_s, 4)
        segments.append(Segment(start_s=start_s, end_s=end_s, source=source))

    return segments


def merge_segments(
    segments: List[Segment],
    merge_gap_ms: int = 250
) -> List[Segment]:
    """
    Merges adjacent segments if the silence gap between them is <= merge_gap_ms.
    
    Args:
        segments: List of Segments (assumed sorted by start_s).
        merge_gap_ms: Maximum gap in ms to bridge between segments (default 250ms).
        
    Returns:
        List of merged Segments.
    """
    if not segments:
        return []

    # Sort segments by start_s
    sorted_segments = sorted(segments, key=lambda s: (s.start_s, s.end_s))
    merge_gap_s = merge_gap_ms / 1000.0
    merged: List[Segment] = []

    current = Segment(
        start_s=sorted_segments[0].start_s,
        end_s=sorted_segments[0].end_s,
        source=sorted_segments[0].source,
        confidence=sorted_segments[0].confidence
    )

    for next_seg in sorted_segments[1:]:
        # If overlapping or gap is within merge_gap_s
        if next_seg.start_s - current.end_s <= merge_gap_s:
            # Extend current segment end if next segment ends later
            current.end_s = max(current.end_s, next_seg.end_s)
        else:
            merged.append(current)
            current = Segment(
                start_s=next_seg.start_s,
                end_s=next_seg.end_s,
                source=next_seg.source,
                confidence=next_seg.confidence
            )

    merged.append(current)
    return merged


def pad_and_clamp_segments(
    segments: List[Segment],
    pad_ms: int = 200,
    total_duration_s: Optional[float] = None
) -> List[Segment]:
    """
    Adds padding before and after each segment to protect word boundaries (chống cắt viền),
    clamps boundaries within [0.0, total_duration_s], and merges any segments that overlap
    as a result of padding expansion.
    
    Args:
        segments: List of Segments.
        pad_ms: Padding in ms to add to both start and end (default 200ms).
        total_duration_s: Optional total audio duration in seconds for upper clamping.
        
    Returns:
        List of padded, clamped, and non-overlapping Segments.
    """
    if not segments:
        return []

    pad_s = pad_ms / 1000.0
    padded_segments: List[Segment] = []

    for seg in segments:
        new_start = max(0.0, round(seg.start_s - pad_s, 4))
        new_end = round(seg.end_s + pad_s, 4)
        if total_duration_s is not None:
            new_end = min(round(total_duration_s, 4), new_end)

        # Ensure valid segment
        if new_end > new_start:
            padded_segments.append(
                Segment(
                    start_s=new_start,
                    end_s=new_end,
                    source=seg.source,
                    confidence=seg.confidence
                )
            )

    # Merge any segments that overlap after padding
    return merge_segments(padded_segments, merge_gap_ms=0)


def filter_short_segments(
    segments: List[Segment],
    min_segment_ms: int = 300
) -> List[Segment]:
    """
    Filters out segments with duration strictly less than min_segment_ms.
    
    Args:
        segments: List of Segments.
        min_segment_ms: Minimum duration in ms (default 300ms).
        
    Returns:
        List of filtered Segments.
    """
    min_segment_s = min_segment_ms / 1000.0
    return [seg for seg in segments if seg.duration >= min_segment_s]


def postprocess_segments(
    raw_segments: List[Segment],
    pad_ms: int = 200,
    merge_gap_ms: int = 250,
    min_segment_ms: int = 300,
    total_duration_s: Optional[float] = None
) -> List[Segment]:
    """
    Standardizes the complete postprocessing pipeline for VAD segments:
    1. Merge nearby raw segments (gap <= merge_gap_ms)
    2. Pad start and end with pad_ms and clamp to [0, total_duration_s]
    3. Filter out short false-alarm segments (< min_segment_ms)
    
    Args:
        raw_segments: Initial raw segments from VAD.
        pad_ms: Padding in ms for boundary protection (default 200ms).
        merge_gap_ms: Gap threshold in ms to merge nearby segments (default 250ms).
        min_segment_ms: Minimum segment duration in ms (default 300ms).
        total_duration_s: Total audio duration in seconds.
        
    Returns:
        List of cleaned, padded, non-overlapping Segment objects.
    """
    if not raw_segments:
        return []

    # Step 1: Merge close segments
    merged = merge_segments(raw_segments, merge_gap_ms=merge_gap_ms)

    # Step 2: Pad and clamp boundaries (also merges any collisions created by padding)
    padded = pad_and_clamp_segments(merged, pad_ms=pad_ms, total_duration_s=total_duration_s)

    # Step 3: Filter out short segments
    filtered = filter_short_segments(padded, min_segment_ms=min_segment_ms)

    return filtered
