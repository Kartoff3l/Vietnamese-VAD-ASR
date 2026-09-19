from typing import Tuple, Union
import numpy
from src.audio import frame_audio

def compute_frame_rms(frame: numpy.ndarray) -> float:
    if len(frame) == 0:
        return 0.0
    return float(numpy.sqrt(numpy.mean(frame.astype(numpy.float32)**2)))

def compute_frame_zcr(frame: np.ndarray) -> float:
    if len(frame) <= 1:
        return 0.0
    signs = np.sign(frame)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(signs[1:] - signs[:-1])) / 2.0
    return float(crossings / (len(frame) - 1))

def compute_rms_energy(
    audio: np.ndarray,
    frame_duration_ms: int = 20,
    sample_rate: int = 16000,
    to_db: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    frames = frame_audio(audio, frame_duration_ms=frame_duration_ms, sample_rate=sample_rate)
    rms_list = [compute_frame_rms(f) for f in frames]
    rms_values = np.array(rms_list, dtype=np.float32)
    frame_step_s = frame_duration_ms / 1000.0
    times = np.arange(len(frames)) * frame_step_s + (frame_step_s / 2.0)
    if to_db:
        eps = 1e-12
        rms_values = 20.0 * np.log10(rms_values + eps)
    return times, rms_values

def compute_zcr(
    audio: np.ndarray,
    frame_duration_ms: int = 20,
    sample_rate: int = 16000
) -> Tuple[np.ndarray, np.ndarray]:
    frames = frame_audio(audio, frame_duration_ms=frame_duration_ms, sample_rate=sample_rate)
    zcr_list = [compute_frame_zcr(f) for f in frames]
    zcr_values = np.array(zcr_list, dtype=np.float32)
    frame_step_s = frame_duration_ms / 1000.0
    times = np.arange(len(frames)) * frame_step_s + (frame_step_s / 2.0)
    return times, zcr_values