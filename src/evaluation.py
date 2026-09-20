import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.schemas import Segment


def load_boundary_labels(json_path: Path) -> List[Segment]:
    """Loads reference speech boundary intervals from a JSON file."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Boundary label file not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return [Segment.from_dict({**item, "source": "reference"}) for item in data]


def segments_to_binary_mask(
    segments: List[Segment],
    total_duration_s: float,
    resolution_ms: int = 10
) -> np.ndarray:
    """
    Converts a list of Segment objects into a fine-grained binary time mask.
    Resolution defaults to 10ms (100 steps per second).
    """
    step_s = resolution_ms / 1000.0
    num_steps = max(1, int(np.ceil(total_duration_s / step_s)))
    mask = np.zeros(num_steps, dtype=bool)

    for seg in segments:
        start_idx = max(0, int(np.floor(seg.start_s / step_s)))
        end_idx = min(num_steps, int(np.ceil(seg.end_s / step_s)))
        mask[start_idx:end_idx] = True

    return mask


def compute_vad_metrics(
    pred_segments: List[Segment],
    ref_segments: List[Segment],
    total_duration_s: float,
    resolution_ms: int = 10
) -> Dict[str, Any]:
    """
    Computes rigorous VAD evaluation metrics comparing predictions against Ground Truth:
    - True Positives (TP seconds)
    - False Positives (FP seconds - False Alarm)
    - False Negatives (FN seconds - Missed Speech)
    - True Negatives (TN seconds)
    - Precision, Recall, F1 Score
    - Intersection-over-Union (IoU)
    - Segment-level boundary errors (start error, end error in ms)
    - Error diagnosis notes
    """
    step_s = resolution_ms / 1000.0
    pred_mask = segments_to_binary_mask(pred_segments, total_duration_s, resolution_ms)
    ref_mask = segments_to_binary_mask(ref_segments, total_duration_s, resolution_ms)

    # Time-level statistics
    tp_steps = np.sum(pred_mask & ref_mask)
    fp_steps = np.sum(pred_mask & (~ref_mask))
    fn_steps = np.sum((~pred_mask) & ref_mask)
    tn_steps = np.sum((~pred_mask) & (~ref_mask))

    tp_sec = round(float(tp_steps * step_s), 4)
    fp_sec = round(float(fp_steps * step_s), 4)
    fn_sec = round(float(fn_steps * step_s), 4)
    tn_sec = round(float(tn_steps * step_s), 4)
    ref_speech_sec = round(float(np.sum(ref_mask) * step_s), 4)
    pred_speech_sec = round(float(np.sum(pred_mask) * step_s), 4)

    precision = round(tp_sec / pred_speech_sec, 4) if pred_speech_sec > 0 else 1.0
    recall = round(tp_sec / ref_speech_sec, 4) if ref_speech_sec > 0 else 1.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    union_sec = round(float(np.sum(pred_mask | ref_mask) * step_s), 4)
    iou = round(tp_sec / union_sec, 4) if union_sec > 0 else 1.0

    # Segment-level boundary matching
    start_errors_ms = []
    end_errors_ms = []
    diagnoses = []

    for ref in ref_segments:
        # Find overlapping or nearest predicted segment
        overlapping_preds = [
            p for p in pred_segments
            if not (p.end_s <= ref.start_s or p.start_s >= ref.end_s)
        ]

        if overlapping_preds:
            # Matched segment span
            matched_start = min(p.start_s for p in overlapping_preds)
            matched_end = max(p.end_s for p in overlapping_preds)
            start_err = abs(matched_start - ref.start_s) * 1000.0  # ms
            end_err = abs(matched_end - ref.end_s) * 1000.0      # ms
            start_errors_ms.append(start_err)
            end_errors_ms.append(end_err)

            if matched_start > ref.start_s + 0.10:
                diagnoses.append(f"Initial clipping (+{(matched_start - ref.start_s)*1000:.0f}ms)")
            if matched_end < ref.end_s - 0.10:
                diagnoses.append(f"Final clipping ({(matched_end - ref.end_s)*1000:.0f}ms)")
        else:
            diagnoses.append(f"Missed speech segment [{ref.start_s:.2f}s - {ref.end_s:.2f}s]")

    # Check for pure false alarms (prediction with zero reference speech)
    for pred in pred_segments:
        has_overlap = any(
            not (pred.end_s <= r.start_s or pred.start_s >= r.end_s)
            for r in ref_segments
        )
        if not has_overlap and pred.duration >= 0.20:
            diagnoses.append(f"False activation at [{pred.start_s:.2f}s - {pred.end_s:.2f}s]")

    mean_start_err_ms = round(float(np.mean(start_errors_ms)), 2) if start_errors_ms else 0.0
    mean_end_err_ms = round(float(np.mean(end_errors_ms)), 2) if end_errors_ms else 0.0

    return {
        "total_duration_s": total_duration_s,
        "ref_speech_s": ref_speech_sec,
        "pred_speech_s": pred_speech_sec,
        "tp_seconds": tp_sec,
        "fp_seconds": fp_sec,
        "fn_seconds": fn_sec,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "mean_start_error_ms": mean_start_err_ms,
        "mean_end_error_ms": mean_end_err_ms,
        "diagnoses": diagnoses if diagnoses else ["Accurate boundary match"]
    }
