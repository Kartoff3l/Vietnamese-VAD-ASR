#temporary runner for ASR
import json
from pathlib import Path
from src.asr import transcribe_full_audio

output_path = Path("result/baseline/test_full_audio.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

result = transcribe_full_audio(
    audio_path="data/eval/testingASR.wav",
    model_size="base",
    language="vi",
    compute_type="int8",
)

output_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Saved: {output_path}")
print(f"Transcript: {result['transcript']}")
print(f"ASR RTF: {result['asr_rtf']}")