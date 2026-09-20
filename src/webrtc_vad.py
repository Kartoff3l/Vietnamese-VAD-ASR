from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional
import numpy as np
import webrtcvad

from src.audio import load_and_resample, float_to_pcm16, frame_audio
from src.schemas import Segment
from src.segmenter import frames_to_segments, postprocess_segments


class WebRTCVAD:
    """
    DSP-aware Voice Activity Detection using WebRTC VAD.
    Operates on 16-bit PCM mono audio framed at 10, 20, or 30 ms (default 20 ms).
    """

    VALID_MODES = (0, 1, 2, 3)
    VALID_FRAME_DURATIONS_MS = (10, 20, 30)
    VALID_SAMPLE_RATES = (8000, 16000, 32000, 48000)

    def __init__(
        self,
        mode: int = 2,
        frame_duration_ms: int = 20,
        sample_rate: int = 16000
    ):
        """
        Initialize WebRTC VAD.

        Args:
            mode: Aggressiveness mode:
                  0: Least aggressive (filters least noise, retains most speech)
                  1: Low aggressiveness
                  2: Moderate aggressiveness (recommended default)
                  3: High aggressiveness (aggressive noise filtering, may clip quiet speech)
            frame_duration_ms: Frame length in ms (10, 20, or 30; default 20 ms).
            sample_rate: Audio sample rate in Hz (default 16000 Hz).
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode {mode}. Must be one of {self.VALID_MODES}.")
        if frame_duration_ms not in self.VALID_FRAME_DURATIONS_MS:
            raise ValueError(
                f"Invalid frame duration {frame_duration_ms} ms. Must be one of {self.VALID_FRAME_DURATIONS_MS}."
            )
        if sample_rate not in self.VALID_SAMPLE_RATES:
            raise ValueError(
                f"Invalid sample rate {sample_rate} Hz. Must be one of {self.VALID_SAMPLE_RATES}."
            )

        self.mode = mode
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * (frame_duration_ms / 1000.0))
        self.frame_bytes_len = self.frame_samples * 2  # 16-bit PCM = 2 bytes per sample

        self.vad = webrtcvad.Vad(self.mode)

    def is_speech(self, frame_bytes: bytes) -> bool:
        """
        Checks if a single PCM16 frame contains speech.

        Args:
            frame_bytes: 16-bit mono PCM bytes of exact expected length.

        Returns:
            True if speech is detected, False otherwise.
        """
        if len(frame_bytes) != self.frame_bytes_len:
            raise ValueError(
                f"Frame size mismatch: expected {self.frame_bytes_len} bytes, got {len(frame_bytes)} bytes."
            )
        return bool(self.vad.is_speech(frame_bytes, self.sample_rate))

    def process_frames(self, frames: List[bytes]) -> List[bool]:
        """
        Processes a list of PCM16 frames and returns boolean speech decisions.

        Args:
            frames: List of PCM16 byte frames.

        Returns:
            List of boolean values where True indicates speech.
        """
        return [self.is_speech(f) for f in frames]

    def detect_speech_segments(
        self,
        audio: Union[str, Path, np.ndarray, bytes],
        pad_ms: int = 200,
        merge_gap_ms: int = 250,
        min_segment_ms: int = 300
    ) -> Tuple[List[Segment], Dict[str, Any]]:
        """
        Full end-to-end speech segmentation pipeline:
        Audio -> 20ms Framing -> WebRTC Frame Decisions -> Raw Segments -> Segment Postprocessing.

        Args:
            audio: File path, float32 numpy array, or PCM16 bytes.
            pad_ms: Padding in ms to add before/after speech to protect boundaries (default 200ms).
            merge_gap_ms: Gap threshold in ms to bridge between close segments (default 250ms).
            min_segment_ms: Minimum duration in ms for a segment to be kept (default 300ms).

        Returns:
            Tuple of (final_segments, statistics_dict)
        """
        # Step 1: Standardize audio into PCM16 bytes and determine total duration
        if isinstance(audio, (str, Path)):
            audio_float, sr = load_and_resample(audio, target_sr=self.sample_rate)
            pcm_bytes = float_to_pcm16(audio_float)
            total_duration_s = round(len(audio_float) / float(self.sample_rate), 4)
        elif isinstance(audio, np.ndarray):
            pcm_bytes = float_to_pcm16(audio)
            total_duration_s = round(len(audio) / float(self.sample_rate), 4)
        elif isinstance(audio, (bytes, bytearray)):
            pcm_bytes = bytes(audio)
            total_samples = len(pcm_bytes) // 2
            total_duration_s = round(total_samples / float(self.sample_rate), 4)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        if len(pcm_bytes) == 0:
            return [], {
                "audio_duration_seconds": 0.0,
                "speech_duration_retained_seconds": 0.0,
                "speech_retained_ratio": 0.0,
                "total_frames": 0,
                "speech_frames": 0,
                "raw_segments_count": 0,
                "final_segments_count": 0,
                "aggressiveness_mode": self.mode
            }

        # Step 2: Frame audio into 20ms PCM16 frames
        frames = frame_audio(
            pcm_bytes,
            frame_duration_ms=self.frame_duration_ms,
            sample_rate=self.sample_rate,
            pad_last_frame=True
        )

        # Step 3: Run WebRTC VAD on each frame
        frame_decisions = self.process_frames(frames)
        speech_frame_count = sum(frame_decisions)

        # Step 4: Convert frame decisions to raw segments
        raw_segments = frames_to_segments(
            frame_decisions,
            frame_duration_ms=self.frame_duration_ms,
            source=f"webrtc_mode_{self.mode}"
        )

        # Step 5: Postprocess segments (merge, padding, boundary clamping, short filtering)
        final_segments = postprocess_segments(
            raw_segments,
            pad_ms=pad_ms,
            merge_gap_ms=merge_gap_ms,
            min_segment_ms=min_segment_ms,
            total_duration_s=total_duration_s
        )

        # Step 6: Compute diagnostic statistics
        retained_duration_s = round(sum(seg.duration for seg in final_segments), 4)
        retained_ratio = round(retained_duration_s / total_duration_s, 4) if total_duration_s > 0 else 0.0

        stats = {
            "audio_duration_seconds": total_duration_s,
            "speech_duration_retained_seconds": retained_duration_s,
            "speech_retained_ratio": retained_ratio,
            "total_frames": len(frames),
            "speech_frames": speech_frame_count,
            "raw_segments_count": len(raw_segments),
            "final_segments_count": len(final_segments),
            "aggressiveness_mode": self.mode
        }

        return final_segments, stats


def run_webrtc_vad(
    audio: Union[str, Path, np.ndarray, bytes],
    mode: int = 2,
    frame_duration_ms: int = 20,
    sample_rate: int = 16000,
    pad_ms: int = 200,
    merge_gap_ms: int = 250,
    min_segment_ms: int = 300
) -> Tuple[List[Segment], Dict[str, Any]]:
    """
    Convenience function to run WebRTC VAD segmentation in a single call.
    """
    vad = WebRTCVAD(mode=mode, frame_duration_ms=frame_duration_ms, sample_rate=sample_rate)
    return vad.detect_speech_segments(
        audio,
        pad_ms=pad_ms,
        merge_gap_ms=merge_gap_ms,
        min_segment_ms=min_segment_ms
    )
