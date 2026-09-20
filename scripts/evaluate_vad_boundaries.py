import sys
import csv
import argparse
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import load_and_resample, float_to_pcm16
from src.webrtc_vad import WebRTCVAD
from src.evaluation import load_boundary_labels, compute_vad_metrics


def run_boundary_evaluation(manifest_path: Path, vad_mode: int = 2):
    print("=" * 110)
    print(f"AUTOMATED VAD BOUNDARY & FALSE POSITIVE / NEGATIVE EVALUATION")
    print(f"Manifest: {manifest_path} | WebRTC VAD Mode: {vad_mode}")
    print("=" * 110)

    if not manifest_path.exists():
        print(f"[ERROR] Manifest file not found at {manifest_path}")
        sys.exit(1)

    annotated_rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            boundary_path_str = row.get("boundary_label_path", "").strip()
            if boundary_path_str:
                annotated_rows.append(row)

    if not annotated_rows:
        print("[WARN] No annotated clips found with boundary_label_path in manifest.")
        return

    print(f"Found {len(annotated_rows)} clips with manual ground-truth boundaries.\n")

    summary_rows = []

    for row in annotated_rows:
        clip_id = row["id"]
        audio_rel_path = row["audio_path"]
        boundary_rel_path = row["boundary_label_path"]
        condition = row.get("condition", "unknown")

        audio_path = PROJECT_ROOT / audio_rel_path
        boundary_path = PROJECT_ROOT / boundary_rel_path

        if not audio_path.exists():
            print(f"[Skip] Audio not found: {audio_path}")
            continue
        if not boundary_path.exists():
            print(f"[Skip] Boundary JSON not found: {boundary_path}")
            continue

        # Load audio & reference
        audio_float, sr = load_and_resample(audio_path, target_sr=16000)
        pcm_bytes = float_to_pcm16(audio_float)
        total_duration = len(audio_float) / float(sr)
        ref_segments = load_boundary_labels(boundary_path)

        # Run WebRTC VAD
        vad = WebRTCVAD(mode=vad_mode, frame_duration_ms=20, sample_rate=sr)
        pred_segments, _ = vad.detect_speech_segments(
            pcm_bytes,
            pad_ms=200,
            merge_gap_ms=250,
            min_segment_ms=300
        )

        # Compute Metrics
        metrics = compute_vad_metrics(pred_segments, ref_segments, total_duration)

        summary_rows.append({
            "id": clip_id,
            "condition": condition,
            "duration": f"{total_duration:.2f}s",
            "ref_speech": f"{metrics['ref_speech_s']:.2f}s",
            "pred_speech": f"{metrics['pred_speech_s']:.2f}s",
            "fp_sec": f"{metrics['fp_seconds']:.2f}s",
            "fn_sec": f"{metrics['fn_seconds']:.2f}s",
            "iou": f"{metrics['iou']:.3f}",
            "recall": f"{metrics['recall']:.3f}",
            "precision": f"{metrics['precision']:.3f}",
            "f1": f"{metrics['f1_score']:.3f}",
            "diagnoses": metrics["diagnoses"]
        })

    # Print Table Header
    header_fmt = "{:<16} | {:<15} | {:<9} | {:<11} | {:<12} | {:<10} | {:<10} | {:<6} | {:<7} | {:<9}"
    print(header_fmt.format("ID", "Condition", "Duration", "Ref Speech", "Pred Speech", "FP (Alarm)", "FN (Miss)", "IoU", "Recall", "Precision"))
    print("-" * 110)

    for r in summary_rows:
        print(header_fmt.format(
            r["id"], r["condition"], r["duration"], r["ref_speech"], r["pred_speech"],
            r["fp_sec"], r["fn_sec"], r["iou"], r["recall"], r["precision"]
        ))

    print("\n" + "-" * 110)
    print("DETAILED ERROR DIAGNOSIS PER CLIP:")
    print("-" * 110)
    for r in summary_rows:
        print(f" * [{r['id']}] ({r['condition']}): IoU={r['iou']} | FP={r['fp_sec']} | FN={r['fn_sec']}")
        print(f"   Diagnosis notes: {'; '.join(r['diagnoses'])}\n")

    print("=" * 110)
    print("EVALUATION FINISHED SUCCESSFULLY!")
    print("=" * 110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VAD speech boundaries against manual annotations.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "test_manifest.csv",
        help="Path to manifest CSV."
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=2,
        help="WebRTC VAD Aggressiveness mode (1, 2, or 3)."
    )
    args = parser.parse_args()
    run_boundary_evaluation(args.manifest, vad_mode=args.mode)
