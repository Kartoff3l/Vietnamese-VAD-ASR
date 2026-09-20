import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Chế độ lưu file không cần màn hình GUI
import matplotlib.pyplot as plt

# Thêm thư mục gốc vào sys.path để import các module trong src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import load_and_resample, float_to_pcm16
from src.features import compute_rms_energy
from src.webrtc_vad import WebRTCVAD
from src.schemas import Segment


def load_ground_truth(boundary_file: Optional[Path]) -> List[Segment]:
    """Đọc file nhãn ranh giới thủ công (.json) nếu có."""
    if not boundary_file or not boundary_file.exists():
        return []
    with open(boundary_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Segment(start_s=item["start_s"], end_s=item["end_s"], source="ground_truth") for item in data]


def generate_spectrogram_rms_vad_plot(
    audio_path: Path,
    boundary_path: Optional[Path],
    output_path: Path,
    title_suffix: str = ""
):
    print(f"\n[DSP Plotting] Đang xử lý: {audio_path.name} ...")
    
    # 1. Nạp và chuẩn hóa âm thanh về 16 kHz Mono
    audio_float, sr = load_and_resample(audio_path, target_sr=16000)
    total_duration_s = len(audio_float) / float(sr)
    time_waveform = np.linspace(0, total_duration_s, len(audio_float))
    pcm_bytes = float_to_pcm16(audio_float)

    # 2. Đọc nhãn Ground Truth (nếu có)
    gt_segments = load_ground_truth(boundary_path)

    # 3. Trích xuất đặc trưng DSP: Short-time RMS Energy (khung 20ms)
    times_rms, rms_vals = compute_rms_energy(audio_float, frame_duration_ms=20, sample_rate=sr)

    # 4. Chạy WebRTC VAD cho 3 mức Aggressiveness (Mode 1, 2, 3)
    vad_results: Dict[int, Any] = {}
    for mode in (1, 2, 3):
        vad = WebRTCVAD(mode=mode, frame_duration_ms=20, sample_rate=sr)
        segments, stats = vad.detect_speech_segments(
            pcm_bytes,
            pad_ms=200,
            merge_gap_ms=250,
            min_segment_ms=300
        )
        vad_results[mode] = (segments, stats)

    # 5. Khởi tạo bố cục biểu đồ 4 tầng
    fig, axes = plt.subplots(
        4, 1,
        figsize=(13, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 2.2, 1.6, 2.2]}
    )
    title_text = f"VietSpeech DSP Diagnostics: {audio_path.name}"
    if title_suffix:
        title_text += f" ({title_suffix})"
    fig.suptitle(title_text, fontsize=14, fontweight="bold", y=0.99)

    # --- PANEL 1: Waveform (Miền thời gian) ---
    ax0 = axes[0]
    ax0.plot(time_waveform, audio_float, color="#2c3e50", linewidth=0.5, label="Waveform (16 kHz)")
    # Tô bóng vùng Ground Truth
    for i, seg in enumerate(gt_segments):
        ax0.axvspan(seg.start_s, seg.end_s, color="#27ae60", alpha=0.25, label="Ground Truth Speech" if i == 0 else "")
    ax0.set_ylabel("Amplitude", fontweight="bold")
    ax0.set_title("1. Waveform (Time Domain) with Ground Truth Annotations", fontsize=11, fontweight="bold")
    ax0.set_ylim(-1.05, 1.05)
    ax0.grid(True, linestyle="--", alpha=0.4)
    ax0.legend(loc="upper right", framealpha=0.9)

    # --- PANEL 2: Spectrogram (Miền tần số STFT) ---
    ax1 = axes[1]
    # NFFT=512, noverlap=384 tương ứng khung phân tích ~32ms, bước nhảy 8ms tại 16kHz
    Pxx, freqs, bins, im = ax1.specgram(
        audio_float,
        NFFT=512,
        Fs=sr,
        noverlap=384,
        cmap="magma"
    )
    ax1.set_ylabel("Frequency (Hz)", fontweight="bold")
    ax1.set_title("2. Spectrogram (STFT Magnitude Spectrum: 0 - 8000 Hz)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 8000)
    # Thêm Colorbar phụ bên cạnh Spectrogram
    cbar = fig.colorbar(im, ax=ax1, pad=0.015, aspect=15)
    cbar.set_label("Power (dB)", fontsize=9)

    # --- PANEL 3: Short-time RMS Energy (20ms frames) ---
    ax2 = axes[2]
    ax2.plot(times_rms, rms_vals, color="#d35400", linewidth=1.2, label="Short-time RMS Energy (20ms)")
    mean_rms = float(np.mean(rms_vals))
    ax2.axhline(mean_rms, color="#e67e22", linestyle=":", linewidth=1.5, label=f"Mean RMS ({mean_rms:.4f})")
    # Đè vùng VAD Mode 2 để tiện so sánh năng lượng với quyết định VAD
    mode2_segs, _ = vad_results[2]
    for seg in mode2_segs:
        ax2.axvspan(seg.start_s, seg.end_s, color="#3498db", alpha=0.15)
    ax2.set_ylabel("RMS Energy", fontweight="bold")
    ax2.set_title("3. Short-Time Energy & Noise Floor Level", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", framealpha=0.9)

    # --- PANEL 4: VAD Speech Intervals Comparison ---
    ax3 = axes[3]
    y_labels = []
    y_positions = []
    colors = {
        "gt": "#27ae60",
        1: "#1abc9c",
        2: "#2980b9",
        3: "#c0392b"
    }

    current_y = 1
    # 4.1 Vẽ Ground Truth (nếu có)
    if gt_segments:
        for seg in gt_segments:
            ax3.broken_barh([(seg.start_s, seg.duration)], (current_y - 0.3, 0.6), facecolors=colors["gt"], edgecolor="#1e8449", alpha=0.9)
        gt_dur = sum(s.duration for s in gt_segments)
        ax3.text(total_duration_s * 1.01, current_y, f"{gt_dur/total_duration_s*100:.1f}% (Ground Truth)", verticalalignment="center", fontsize=9, fontweight="bold", color=colors["gt"])
        y_labels.append("Ground Truth")
        y_positions.append(current_y)
        current_y += 1

    # 4.2 Vẽ WebRTC VAD Modes (1, 2, 3)
    for mode in [1, 2, 3]:
        segs, stats = vad_results[mode]
        for seg in segs:
            ax3.broken_barh([(seg.start_s, seg.duration)], (current_y - 0.3, 0.6), facecolors=colors[mode], edgecolor="#2c3e50", alpha=0.85)
        retained_pct = stats["speech_retained_ratio"] * 100
        ax3.text(
            total_duration_s * 1.01,
            current_y,
            f"{retained_pct:.1f}% ({stats['final_segments_count']} segs)",
            verticalalignment="center",
            fontsize=9,
            fontweight="bold",
            color=colors[mode]
        )
        mode_desc = "Mode 1 (Low)" if mode == 1 else ("Mode 2 (Default)" if mode == 2 else "Mode 3 (High)")
        y_labels.append(mode_desc)
        y_positions.append(current_y)
        current_y += 1

    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(y_labels, fontweight="bold")
    ax3.set_xlabel("Time (seconds)", fontweight="bold")
    ax3.set_ylabel("Decision", fontweight="bold")
    ax3.set_title("4. VAD Speech Segments by Mode (200ms pad, 250ms merge)", fontsize=11, fontweight="bold")
    ax3.set_xlim(0, total_duration_s * 1.18)
    ax3.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=250, bbox_inches="tight")
    plt.close()
    print(f"[THÀNH CÔNG] Đã lưu biểu đồ tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tạo biểu đồ Spectrogram/RMS/VAD cho 2 trường hợp đại diện.")
    parser.add_argument("--all-cases", action="store_true", help="Tự động tạo cả 2 trường hợp đại diện.")
    args = parser.parse_args()

    # Đường dẫn 2 trường hợp đại diện theo Kế hoạch Dự án
    case_clean = {
        "audio": PROJECT_ROOT / "data" / "eval" / "clean" / "VIVOSDEV07_107.wav",
        "boundary": PROJECT_ROOT / "data" / "boundaries" / "VIVOSDEV07_107.json",
        "output": PROJECT_ROOT / "results" / "figures" / "case1_clean_speech_vad.png",
        "title": "Case 1: Clean Speech with Pauses"
    }

    case_noise = {
        "audio": PROJECT_ROOT / "data" / "eval" / "moderate_noise" / "bud500_1.wav",
        "boundary": PROJECT_ROOT / "data" / "boundaries" / "bud500_1.json",
        "output": PROJECT_ROOT / "results" / "figures" / "case2_moderate_noise_vad.png",
        "title": "Case 2: Moderate Noise Speech"
    }

    # Chạy xuất ảnh
    generate_spectrogram_rms_vad_plot(case_clean["audio"], case_clean["boundary"], case_clean["output"], case_clean["title"])
    generate_spectrogram_rms_vad_plot(case_noise["audio"], case_noise["boundary"], case_noise["output"], case_noise["title"])


if __name__ == "__main__":
    main()  huo