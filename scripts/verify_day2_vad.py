import sys
import os
import argparse
from pathlib import Path

# Fix Windows console UTF-8 output if possible
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import inspect_audio, load_and_resample, float_to_pcm16, frame_audio
from src.features import compute_rms_energy, compute_zcr
from src.webrtc_vad import WebRTCVAD


def verify_day2_webrtc_vad(audio_path: Path):
    print("=" * 80)
    print("VIETSPEECH DAY 2 VERIFICATION: WebRTC VAD & Segment Postprocessing")
    print(f"Audio File: {audio_path}")
    print("=" * 80)

    if not audio_path.exists():
        print(f"[ERROR] File not found at {audio_path}")
        sys.exit(1)

    # 1. Audio Inspection
    info = inspect_audio(audio_path)
    print("\n[1] AUDIO INSPECTION:")
    print(f"   - File name:         {info['file_name']}")
    print(f"   - Original SR:       {info['sample_rate']} Hz")
    print(f"   - Channels:          {info['channels']}")
    print(f"   - Duration:          {info['duration_seconds']} seconds")
    print(f"   - Subtype / Format:  {info['subtype']} / {info['format']}")

    # 2. Audio Loading & Resampling to 16 kHz mono
    audio_float, sr = load_and_resample(audio_path, target_sr=16000)
    pcm_bytes = float_to_pcm16(audio_float)
    total_duration = round(len(audio_float) / float(sr), 4)

    # 3. DSP Framing & Features
    times_rms, rms_vals = compute_rms_energy(audio_float, frame_duration_ms=20, sample_rate=sr)
    times_zcr, zcr_vals = compute_zcr(audio_float, frame_duration_ms=20, sample_rate=sr)
    frames = frame_audio(pcm_bytes, frame_duration_ms=20, sample_rate=sr)

    print("\n[2] DSP 20ms FRAMING & SHORT-TIME ANALYSIS:")
    print(f"   - Standardized SR:   {sr} Hz (Nyquist: {sr // 2} Hz)")
    print(f"   - Frame Duration:    20 ms ({int(sr * 0.02)} samples = 640 bytes PCM16)")
    print(f"   - Total 20ms Frames: {len(frames)} frames")
    print(f"   - RMS Energy Range:  {rms_vals.min():.5f} -> {rms_vals.max():.5f} (Mean: {rms_vals.mean():.5f})")
    print(f"   - ZCR Range:         {zcr_vals.min():.5f} -> {zcr_vals.max():.5f} (Mean: {zcr_vals.mean():.5f})")

    # 4. WebRTC VAD Across Modes 1, 2, 3
    print("\n[3] WebRTC VAD ACROSS AGGRESSIVENESS MODES (1, 2, 3):")
    print("-" * 80)
    print(f"{'Mode':<8} | {'Speech Frames':<15} | {'Raw Segs':<10} | {'Final Segs':<12} | {'Speech Retained':<18} | {'Retained %':<10}")
    print("-" * 80)

    results_by_mode = {}
    for mode in (1, 2, 3):
        vad = WebRTCVAD(mode=mode, frame_duration_ms=20, sample_rate=sr)
        segments, stats = vad.detect_speech_segments(
            pcm_bytes,
            pad_ms=200,
            merge_gap_ms=250,
            min_segment_ms=300
        )
        results_by_mode[mode] = (segments, stats)
        speech_frames_str = f"{stats['speech_frames']}/{stats['total_frames']}"
        retained_str = f"{stats['speech_duration_retained_seconds']:.2f}s / {stats['audio_duration_seconds']:.2f}s"
        pct_str = f"{stats['speech_retained_ratio'] * 100:.1f}%"
        print(f"Mode {mode:<3} | {speech_frames_str:<15} | {stats['raw_segments_count']:<10} | {stats['final_segments_count']:<12} | {retained_str:<18} | {pct_str:<10}")

    print("-" * 80)

    # 5. Detailed Segment Breakdown for Default Mode 2
    default_segments, default_stats = results_by_mode[2]
    print("\n[4] SEGMENT BREAKDOWN (Default Mode 2, pad=200ms, merge_gap=250ms, min_duration=300ms):")
    if not default_segments:
        print("   (No speech segments detected)")
    else:
        for idx, seg in enumerate(default_segments, 1):
            print(f"   Segment #{idx:02d}: [{seg.start_s:6.2f}s -> {seg.end_s:6.2f}s] | Duration: {seg.duration:5.2f}s | Source: {seg.source}")

    # 6. Boundary Protection & Clamping Check
    print("\n[5] BOUNDARY PROTECTION (CHONG CAT VIEN) VERIFICATION:")
    all_valid = True
    for idx, seg in enumerate(default_segments, 1):
        if seg.start_s < 0.0:
            print(f"   [FAIL] Segment #{idx} start_s < 0.0 ({seg.start_s})")
            all_valid = False
        if seg.end_s > total_duration:
            print(f"   [FAIL] Segment #{idx} end_s > total duration ({seg.end_s} > {total_duration})")
            all_valid = False
        if seg.end_s <= seg.start_s:
            print(f"   [FAIL] Segment #{idx} end_s <= start_s ({seg.end_s} <= {seg.start_s})")
            all_valid = False

    # Check non-overlapping
    for i in range(len(default_segments) - 1):
        if default_segments[i].end_s > default_segments[i+1].start_s:
            print(f"   [FAIL] Overlap detected between Segment #{i+1} and #{i+2}")
            all_valid = False

    if all_valid:
        print("   [PASS] Boundary protection and clamping PASSED (no negative starts, no duration overflows, non-overlapping).")

    print("\n" + "=" * 80)
    print("DAY 2 WebRTC VAD & SEGMENTER VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Day 2 WebRTC VAD and Segmenter logic.")
    parser.add_argument(
        "--audio",
        type=Path,
        default=PROJECT_ROOT / "data" / "demo" / "test_sample_day2.wav",
        help="Path to audio file for testing."
    )
    args = parser.parse_args()
    verify_day2_webrtc_vad(args.audio)
