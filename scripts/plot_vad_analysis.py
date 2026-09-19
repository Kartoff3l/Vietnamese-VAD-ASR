import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import load_and_resample, float_to_pcm16
from src.features import compute_rms_energy, compute_zcr
from src.webrtc_vad import WebRTCVAD


def plot_vad_dsp_analysis(audio_path: Path, output_path: Path):
    print(f"Generating DSP & WebRTC VAD analysis plot for {audio_path} ...")
    
    # 1. Load and resample audio
    audio_float, sr = load_and_resample(audio_path, target_sr=16000)
    total_duration_s = len(audio_float) / float(sr)
    time_waveform = np.linspace(0, total_duration_s, len(audio_float))
    pcm_bytes = float_to_pcm16(audio_float)

    # 2. Extract DSP Features (20ms frames)
    times_rms, rms_vals = compute_rms_energy(audio_float, frame_duration_ms=20, sample_rate=sr)
    times_zcr, zcr_vals = compute_zcr(audio_float, frame_duration_ms=20, sample_rate=sr)

    # 3. Compute VAD segments for modes 1, 2, 3
    vad_results = {}
    for mode in (1, 2, 3):
        vad = WebRTCVAD(mode=mode, frame_duration_ms=20, sample_rate=sr)
        segments, stats = vad.detect_speech_segments(
            pcm_bytes,
            pad_ms=200,
            merge_gap_ms=250,
            min_segment_ms=300
        )
        vad_results[mode] = (segments, stats)

    # 4. Create Plot Figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1.5, 1.5, 2]})
    fig.suptitle(f"VietSpeech DSP & WebRTC VAD Diagnostics: {audio_path.name}", fontsize=14, fontweight="bold")

    # (A) Waveform Plot with Mode 2 Overlay
    ax0 = axes[0]
    ax0.plot(time_waveform, audio_float, color="#2c3e50", linewidth=0.6, label="Waveform (16 kHz mono)")
    # Overlay Mode 2 segments
    mode2_segs, mode2_stats = vad_results[2]
    for i, seg in enumerate(mode2_segs):
        ax0.axvspan(seg.start_s, seg.end_s, color="#2ecc71", alpha=0.3, label="Speech (WebRTC Mode 2)" if i == 0 else "")
    ax0.set_ylabel("Amplitude", fontweight="bold")
    ax0.set_title("Waveform (Time Domain) with WebRTC Mode 2 Speech Boundaries", fontsize=11)
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper right")

    # (B) RMS Energy Plot
    ax1 = axes[1]
    ax1.plot(times_rms, rms_vals, color="#e67e22", linewidth=1.2, label="Short-time RMS Energy (20ms frames)")
    ax1.axhline(rms_vals.mean(), color="#d35400", linestyle=":", label=f"Mean RMS ({rms_vals.mean():.4f})")
    for seg in mode2_segs:
        ax1.axvspan(seg.start_s, seg.end_s, color="#2ecc71", alpha=0.15)
    ax1.set_ylabel("RMS Energy", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # (C) Zero-Crossing Rate Plot
    ax2 = axes[2]
    ax2.plot(times_zcr, zcr_vals, color="#9b59b6", linewidth=1.2, label="Zero-Crossing Rate (ZCR)")
    for seg in mode2_segs:
        ax2.axvspan(seg.start_s, seg.end_s, color="#2ecc71", alpha=0.15)
    ax2.set_ylabel("ZCR", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    # (D) VAD Modes Comparison (Mode 1, 2, 3)
    ax3 = axes[3]
    y_labels = ["Mode 3 (High)", "Mode 2 (Moderate)", "Mode 1 (Low)"]
    y_positions = [3, 2, 1]
    colors = ["#e74c3c", "#3498db", "#1abc9c"]

    for idx, (mode, y_pos, col) in enumerate(zip([3, 2, 1], y_positions, colors)):
        segs, stats = vad_results[mode]
        for seg in segs:
            ax3.broken_barh(
                [(seg.start_s, seg.duration)],
                (y_pos - 0.3, 0.6),
                facecolors=col,
                edgecolor="#2c3e50",
                alpha=0.85
            )
        retained_pct = stats["speech_retained_ratio"] * 100
        ax3.text(
            total_duration_s * 1.01,
            y_pos,
            f"{retained_pct:.1f}% retained ({stats['final_segments_count']} segs)",
            verticalalignment="center",
            fontsize=9,
            fontweight="bold",
            color=col
        )

    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(y_labels, fontweight="bold")
    ax3.set_xlabel("Time (seconds)", fontweight="bold")
    ax3.set_ylabel("VAD Mode", fontweight="bold")
    ax3.set_title("WebRTC VAD Speech Intervals by Aggressiveness Mode (200ms pad, 250ms merge)", fontsize=11)
    ax3.set_xlim(0, total_duration_s * 1.15)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Diagnostic figure saved successfully to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DSP features and WebRTC VAD comparison.")
    parser.add_argument(
        "--audio",
        type=Path,
        default=PROJECT_ROOT / "data" / "demo" / "test_sample_day2.wav",
        help="Path to input audio file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "webrtc_vad_analysis.png",
        help="Path to save output figure."
    )
    args = parser.parse_args()
    plot_vad_dsp_analysis(args.audio, args.output)
