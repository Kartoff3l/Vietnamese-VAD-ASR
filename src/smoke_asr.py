from faster_whisper import WhisperModel

audio_path = "data/eval/testingASR.wav"

model = WhisperModel(
    "base",
    device = "cpu",
    compute_type = "int8",
)

segments, info = model.transcribe(
    audio_path,
    language="vi",
    task="transcribe",
    beam_size=5,
    vad_filter=False,
)

print(f"Detected language: {info.language}")
print(f"Language probability: {info.language_probability:.3f}")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}")
