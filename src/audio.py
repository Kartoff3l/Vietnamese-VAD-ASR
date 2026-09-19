from pathlib import Path
from typing import Dict, Any, Tuple, List, Union
import numpy
import soundfile
import torchaudio

def inspect_audio(file_path: Union[str, Path]) -> Dict[str, Any]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    info = soundfile.info(str(file_path))

    return {
    "file_name": file_path.name,
    "sample_rate": info.samplerate,
    "channels": info.channels,
    "duration_seconds": round(info.duration, 4),
    "total_samples": info.frames,
    "subtype": info.subtype,
    "format": info.format
    }
    
def load_and_resample(
    file_path: Union[str, Path],
    target_sr: int = 16000,
    normalize_peak: bool = False
) -> Tuple[numpy.ndarray, int]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    waveform, orig_sr = torchaudio.load(str(file_path))
    
    if waveform.shape[0] > 1:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr
        )
        waveform = resampler(waveform)
        
    audio_np = waveform.squeeze(0).numpy().astype(numpy.float32)

    if normalize_peak:
        max_val = numpy.max(numpy.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return audio_np, target_sr

def float_to_pcm16(audio_float: numpy.array) -> bytes:
    clipped = numpy.clip(audio_float, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).round().astype(numpy.int16)
    return pcm16.tobytes()

def pcm16_to_float(pcm_bytes: bytes) -> numpy.ndarray:
    pcm16 = numpy.fromgubber(pcm_bytes, dtype=np.int16)
    audio_float = pcm16.astype(np.float32) / 32767.0

    return audio_float

def frame_audio(
    audio: Union[numpy.ndarray, bytes],
    frame_duration_ms: int = 20,
    sample_rate: int = 16000,
    pad_last_frame: bool = True
) -> List[Union[numpy.ndarray, bytes]]:

    frame_samples = int(sample_rate * (frame_duration_ms * 0.02)) # 12000 * 0.02 = 320
    # frame duration = 0.02s = 20ms

    if isinstance(audio, (byates, bytearray)):
        frame_bytes_len = frame_samples * 2 
        total_bytess = len(audio)
        frams = []

        for offset in range(0, total_bytess, frame_bytes_len):
            chunk = audio[offset:offset+frame_bytes_len]
            if len(chunk) == frame_bytes_len:
                frames.append(chunk)
            elif pad_last_frame and len(chunk) > 0:
                # trong trường hợp frame cuối không đủ 20ms thì thêm vào các byte 0x00 để đủ 20ms
                padded_chunk = chunk + b'\x00' * (frame_bytes_len - len(chunk))
                frames.append(padded_chunk)
            return frames

    elif isinstance(audio, np.ndarry):
        total_samples = len(audio)
        frames = []

        for offset in range(0, total_samples, frame_samples):
            chunk = audio[offset:offset+frame_samples]
            if len(chunk) == frame_samples:
                frames.append(chunk)
            elif pad_last_frame and len(chunk) > 0:
                pad_width = frames_samples - len(chunk)
                padded_chunk = numpy.pad(chunk, (0, pad_width), mode='constant', constant_values=0)
                frames.append(padded_chunk)
            return frames
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}. Only numpy arrays or bytes are supported.")