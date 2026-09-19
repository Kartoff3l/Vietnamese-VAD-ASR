# 🎙️ VietSpeech — Final Team Project Plan

> **Vietnamese VAD-Gated ASR Evaluation and API Pipeline**  
> **Duration:** 4 days  
> **Team:** 2 members — Person A (Computer Science) and Person B (Electrical & Computer Engineering)  
> **Scope:** Offline audio-file processing only. No microphone streaming, custom model training, diarization, complex frontend, or Docker in the core scope.

---

## 1. Project Goal

Build a reproducible Vietnamese Automatic Speech Recognition (ASR) pipeline that compares:

1. **Whole-audio ASR** — send the full audio file to ASR.
2. **WebRTC-VAD-gated ASR** — detect speech with a lightweight frame-based VAD, then transcribe only detected speech segments.
3. **Silero-VAD-gated ASR** — detect speech with neural VAD, then transcribe only detected speech segments.

The project evaluates the trade-off between transcription quality, compute efficiency, and speech-boundary quality.

### Core research question

> How do audio preprocessing and VAD choice affect Vietnamese ASR accuracy, offline processing efficiency, and speech-segment boundary quality?

### Why this is portfolio-relevant

The final system demonstrates practical Speech AI skills:

- Vietnamese ASR with Whisper/faster-whisper
- Voice Activity Detection with WebRTC VAD and Silero VAD
- Audio preprocessing and PCM/frame handling
- DSP concepts: sampling rate, framing, RMS energy, ZCR, spectrograms
- Speech-data collection, labelling, and validation
- WER, CER, Real-Time Factor, and VAD error analysis
- FastAPI model-serving endpoint
- Unit testing, reproducibility, Git branches, pull requests, and technical documentation

---

## 2. Final System Design

```text
Vietnamese audio file (.wav/.mp3/.m4a)
                │
                ▼
Audio preprocessing
Decode → mono → resample to 16 kHz → waveform / PCM16
                │
                ├───────────────┬─────────────────┐
                ▼               ▼                 ▼
          No external VAD   WebRTC VAD        Silero VAD
          (full-audio       20 ms PCM frames  neural VAD
           baseline)        modes 1/2/3       threshold-based
                │               │                 │
                └───────────────┴─────────────────┘
                                │
                                ▼
Common segment postprocessor
Merge close segments → add padding → clamp boundaries → filter short segments
                                │
                                ▼
faster-whisper Vietnamese ASR (language="vi")
                                │
                                ▼
Transcript + timestamps + JSON/SRT export
                                │
                                ▼
Evaluation and reporting
WER • CER • ASR RTF • pipeline RTF • ASR audio ratio • segment count • boundary metrics
```

### Core pipeline rule

Both WebRTC VAD and Silero VAD **must use the same segment postprocessing**:

- Same padding
- Same merge-gap rule
- Same minimum segment duration
- Same boundary clamping

This makes the comparison fair: the benchmark compares VAD methods rather than different downstream segmentation logic.

---

## 3. Scope and Deliverables

### Required deliverables

- Python CLI for full-audio, WebRTC VAD, and Silero VAD pipelines.
- One standardized input format: 16 kHz mono audio.
- WebRTC VAD implementation using 20 ms signed PCM16 frames.
- Silero VAD integration.
- faster-whisper ASR backend configured for Vietnamese.
- A manually transcribed Vietnamese evaluation set of 18–24 audio files.
- Manual speech-boundary labels for 6–8 selected clips.
- Benchmark scripts and a CSV of results.
- WER/CER evaluation with Vietnamese text normalization.
- VAD diagnostics: speech ratio, segment count, missed-speech and false-activation examples.
- Minimal FastAPI service:
  - `GET /health`
  - `POST /transcribe`
- Unit tests.
- Architecture diagram, DSP visualizations, benchmark table, error analysis, README, and 30–60 second demo GIF/video.

### Explicitly out of scope

- Live microphone streaming and online endpointing
- Training or fine-tuning Whisper/ASR models
- Speaker diarization or speaker recognition
- TTS and speech enhancement models
- Large web frontend / complex Streamlit interface
- Docker deployment
- Large public datasets or hundreds of manual annotations
- Multiple ASR backends or a broad model-size comparison

### Scope priority if time is limited

1. Correct full-audio / WebRTC / Silero benchmark
2. Labelled Vietnamese test data and WER/CER
3. VAD metrics and DSP visualizations
4. Tests and reproducible CLI
5. FastAPI endpoint
6. Demo GIF/video and extra polish

---

## 4. Dataset Plan

### Evaluation set target

Prepare **18–24 Vietnamese audio clips**, ideally 5–45 seconds each.

| Condition | Target clips | Purpose |
|---|---:|---|
| Clean Vietnamese speech | 5 | Baseline ASR quality |
| Pause-heavy speech | 4 | Silence removal and segmentation benefit |
| Moderate background noise | 4 | Noise robustness and VAD false activations |
| Short utterances | 3 | Short-segment and boundary behavior |
| Technical / Vietnamese–English code-switching | 2–4 | Realistic Vietnamese Speech AI use case |
| **Total** | **18–24** | Fixed test set for all pipelines |

### Manual annotations

For every clip, prepare a reference transcript.

For **6–8 selected clips**, also annotate reference speech segments:

```json
[
  {"start_s": 0.40, "end_s": 3.25},
  {"start_s": 4.10, "end_s": 7.85}
]
```

Select boundary-labelled examples across clean, pause-heavy, noisy, quiet, and short-utterance conditions.

### Dataset manifest

`data/test_manifest.csv`

```csv
id,audio_path,reference_text,condition,speaker_id,duration_seconds,boundary_label_path
001,data/eval/clean_001.wav,"hôm nay tôi đang thử nghiệm hệ thống nhận dạng giọng nói tiếng việt",clean,speaker_01,8.1,
002,data/eval/pause_001.wav,"mình cần kiểm tra độ trễ của mô hình",pause_heavy,speaker_01,12.4,data/boundaries/pause_001.json
```

### Transcript normalization policy

Apply identical normalization to reference and prediction before WER/CER:

- Lowercase text
- Trim surrounding whitespace
- Collapse repeated spaces
- Define one consistent punctuation policy
- Define one consistent policy for digits and spoken numbers
- Preserve Vietnamese diacritics for the main WER/CER score
- Document handling of English technical words such as `API`, `Python`, `Docker`, and `GitHub`

---

## 5. Metrics and Experiment Design

### ASR metrics

| Metric | Definition | Desired direction |
|---|---|---|
| WER | Word Error Rate | Lower |
| CER | Character Error Rate | Lower |
| ASR RTF | ASR runtime / original audio duration | Lower |
| Pipeline RTF | preprocessing + VAD + ASR runtime / original audio duration | Lower |
| Audio passed to ASR | VAD segment duration / original audio duration | Lower, without speech loss |
| Segment count | Number of final VAD segments | Context-dependent |

\[
WER = \frac{S + D + I}{N}
\]

Where \(S\) is substitutions, \(D\) is deletions, \(I\) is insertions, and \(N\) is reference words.

\[
RTF = \frac{\text{processing time}}{\text{original audio duration}}
\]

### VAD boundary metrics

For matched predicted segment \(p=[p_s,p_e]\) and manually labelled reference segment \(r=[r_s,r_e]\):

\[
E_{start} = |p_s-r_s|
\]

\[
E_{end} = |p_e-r_e|
\]

\[
E_{boundary} = \frac{E_{start}+E_{end}}{2}
\]

Also report segment intersection-over-union:

\[
IoU = \frac{\text{duration}(p \cap r)}{\text{duration}(p \cup r)}
\]

### Benchmark configurations

Keep the main comparison controlled:

| Pipeline | VAD settings | ASR settings |
|---|---|---|
| Full-audio + ASR | No external VAD | faster-whisper `base`, `language="vi"` |
| WebRTC VAD + ASR | 20 ms frame, aggressiveness 2 | Same ASR settings |
| Silero VAD + ASR | threshold 0.50 | Same ASR settings |

Run a small VAD parameter study only after the main benchmark:

| VAD method | Parameter values |
|---|---|
| WebRTC VAD | Aggressiveness 1, 2, 3 |
| Silero VAD | Threshold 0.35, 0.50, 0.65 |

Keep the ASR model fixed for the VAD benchmark. Do not compare several Whisper model sizes unless all required work is already complete.

### Final result tables

#### ASR and efficiency table

| Pipeline | Mean WER ↓ | Mean CER ↓ | ASR RTF ↓ | Pipeline RTF ↓ | Audio passed to ASR ↓ |
|---|---:|---:|---:|---:|---:|
| Full-audio + Whisper | — | — | — | — | 100% |
| WebRTC VAD + Whisper | — | — | — | — | —% |
| Silero VAD + Whisper | — | — | — | — | —% |

#### VAD-quality table

| Method | Median start error ↓ | Median end error ↓ | Median IoU ↑ | Missed-speech cases ↓ | False-activation cases ↓ |
|---|---:|---:|---:|---:|---:|
| RMS-energy diagnostic VAD | — ms | — ms | — | — | — |
| WebRTC VAD | — ms | — ms | — | — | — |
| Silero VAD | — ms | — ms | — | — | — |

---

## 6. Technical Stack

### Required software

```text
Python 3.10+
PyTorch + torchaudio
soundfile
webrtcvad-wheels
silero-vad
faster-whisper
jiwer
numpy
pandas
matplotlib
fastapi
uvicorn[standard]
python-multipart
pytest
httpx
pyyaml
```

### Core tools and responsibilities

| Tool | Use in project |
|---|---|
| `torchaudio` / `soundfile` | Load, save, resample, and inspect audio |
| `webrtcvad-wheels` | Frame-based WebRTC VAD baseline |
| `silero-vad` | Neural VAD comparator |
| `faster-whisper` | Vietnamese speech-to-text inference |
| `jiwer` | WER/CER calculation and error analysis |
| `numpy` / `pandas` | Audio arrays, manifests, benchmark CSVs |
| `matplotlib` | RMS, waveform, STFT/spectrogram, VAD overlays |
| `FastAPI` / `uvicorn` | Minimal transcription API |
| `pytest` / `httpx` | Unit and API tests |
| `PyYAML` | Reproducible experiment configuration |

### Initial configuration

`configs/default.yaml`

```yaml
audio:
  target_sample_rate: 16000
  mono: true
  normalize_peak: true

segmentation:
  pad_ms: 200
  merge_gap_ms: 250
  min_segment_ms: 300

webrtc:
  frame_ms: 20
  aggressiveness: 2

silero:
  threshold: 0.50
  min_speech_duration_ms: 250
  min_silence_duration_ms: 300

asr:
  model: base
  language: vi
  task: transcribe
  beam_size: 5
  compute_type: int8
```

---

## 7. Roles and Ownership

### Person A — Computer Science / Speech AI System Owner

**Main responsibility:** Own the end-to-end Speech AI pipeline, ASR integration, evaluation, reproducibility, and deployable service.

#### Person A owns

- Repository architecture and integration workflow
- Common VAD interface and shared segment postprocessing
- faster-whisper integration
- Silero VAD integration
- RMS-energy diagnostic VAD implementation
- Transcript merging, timestamp handling, JSON/SRT output
- Vietnamese text normalization and JiWER evaluation
- WER, CER, ASR RTF, pipeline RTF, benchmark reporting
- Benchmark runner and error analysis
- FastAPI endpoint, request/response schema, API tests
- README architecture, quick start, results, limitations

#### Person A learning outcomes

- Practical ASR inference and Vietnamese evaluation
- Neural VAD integration and VAD/ASR interaction
- Reproducible speech-model benchmarking
- Speech quality metrics and error analysis
- Production-oriented FastAPI model serving
- System integration and testable pipeline design

### Person B — Electrical & Computer Engineering / DSP and Audio Owner

**Main responsibility:** Own the DSP audio frontend, frame-based WebRTC VAD, VAD boundary evaluation, audio data preparation, and diagnostic visualizations.

#### Person B owns

- Audio inspection, decoding checks, resampling, stereo-to-mono conversion
- Signed PCM16 conversion for WebRTC VAD
- 20 ms frame generation and validation
- RMS energy and ZCR feature functions
- WebRTC VAD adapter and aggressiveness experiments
- Frame-label-to-segment conversion
- Validation of segment merge, padding, and clipping behavior
- Audio recording/collection, manual transcripts, and boundary labels
- VAD metrics: audio passed to ASR, segment count, missed speech, false activations, boundary error, IoU
- Waveform/RMS/STFT/spectrogram visualizations with VAD overlays
- DSP/data documentation and audio/VAD tests

#### Person B learning outcomes

- Practical DSP pipeline design for speech
- Sampling, PCM encoding, frame processing, energy analysis, STFT
- Frame-based VAD behavior and noise/speech trade-offs
- Speech-boundary annotation and VAD evaluation
- Audio test-fixture creation and robust preprocessing tests

### Shared responsibilities

- Use feature branches and review each other’s pull requests before merge.
- Pair-program integration at least once daily.
- Freeze the labelled test dataset before final benchmark runs.
- Run final benchmarks together on the same machine/configuration.
- Review difficult VAD and ASR errors together.
- Co-author README, architecture diagram, final result interpretation, and demo.
- Each member must be able to explain the full system and their exact individual contribution.

---

## 8. Four-Day Execution Schedule

## Day 1 — Baseline, Data Format, and DSP Foundation

### Person A — CS

- Create repository structure, Python environment, `requirements.txt`, and configuration file.
- Implement CLI skeleton and full-audio faster-whisper baseline.
- Set `language="vi"`; save transcript, timestamps, runtime, and audio duration to JSON.
- Define output schema and benchmark CSV schema.
- Implement first version of Vietnamese text normalization and JiWER WER/CER wrappers.
- Record preprocessing time, ASR time, and total time separately.

### Person B — ECE

- Implement audio inspection: duration, sample rate, channels, dtype, and basic validation.
- Implement conversion to mono, 16 kHz, and signed PCM16 WAV/bytes.
- Implement 20 ms framing at 16 kHz: 320 samples per frame.
- Implement RMS energy and optional ZCR calculation.
- Record/collect and manually transcribe 8–10 Vietnamese clips.
- Create initial `data/test_manifest.csv` and document annotation rules.
- Generate RMS plots for one clean and one noisy recording.

### Shared checkpoint

- Full-audio Vietnamese ASR works on 3–5 clips.
- All test files can be converted to 16 kHz mono.
- 8–10 clips have verified reference transcripts.
- Manifest schema and transcript-normalization rules are agreed and committed.

### End-of-day deliverables

- Baseline CLI command works.
- Initial labelled dataset exists.
- Audio preprocessing utilities work.
- Two initial DSP figures exist.

---

## Day 2 — VAD Methods and Boundary Diagnostics

### Person A — CS

- Define common VAD segment schema:

```python
@dataclass
class Segment:
    start_s: float
    end_s: float
    source: str
```

- Implement common segment postprocessor: merge, padding, clamp, minimum-duration filtering.
- Implement Silero VAD adapter returning common segments.
- Implement RMS-energy VAD for diagnostic comparison only.
- Add CLI choices: `--vad none|webrtc|silero`.
- Start unit tests for segment merging, padding, and boundary clamping.

### Person B — ECE

- Implement WebRTC VAD wrapper with PCM16 mono audio and 20 ms frames.
- Implement WebRTC aggressiveness modes 1, 2, and 3.
- Convert frame decisions into timestamped segments.
- Manually label speech start/end boundaries for 6–8 selected clips.
- Verify segments by listening to extracted audio clips.
- Create at least one waveform/RMS/VAD-overlay figure.

### Shared checkpoint

- WebRTC and Silero return the same segment JSON format.
- Both systems use the identical shared postprocessor.
- Inspect VAD output visually and by listening.
- Agree default experimental settings: WebRTC mode 2, Silero threshold 0.50, 200 ms padding, 250 ms merge gap.

### End-of-day deliverables

- WebRTC VAD, Silero VAD, and diagnostic energy VAD produce timestamps.
- Boundary labels exist for 6–8 clips.
- Segment output is compatible with downstream ASR.

---

## Day 3 — VAD-Gated ASR and Final Benchmark

### Person A — CS

- Connect VAD segments to faster-whisper inference.
- Implement full-audio, WebRTC-gated, and Silero-gated ASR paths.
- Implement transcript merging with original-audio timestamps.
- Export JSON and optional SRT output.
- Implement WER, CER, ASR RTF, pipeline RTF, runtime, and aggregated benchmark CSV.
- Run benchmark and analyze the five most significant ASR errors.

### Person B — ECE

- Expand/complete the dataset to 18–24 clips.
- Verify every reference transcript and condition label.
- Compute VAD metrics: audio passed to ASR, segment count, missed speech, false activations.
- Implement boundary matching, start/end error, and segment IoU for the 6–8 boundary-labelled clips.
- Generate two representative diagnostic figures: clean/pause-heavy and noisy/quiet speech.

### Shared checkpoint

- Freeze test manifest, transcript labels, and environment settings.
- Run all three pipelines on the same fixed dataset, model, machine, and configuration.
- Record CPU/GPU, RAM, OS, Python version, package versions, model, and compute type.
- Discuss and write the preliminary conclusion using actual results only.

### End-of-day deliverables

- Complete ASR and VAD benchmark outputs.
- Benchmark CSV and two final DSP/VAD figures.
- Error analysis for at least five difficult cases.

---

## Day 4 — API, Tests, Documentation, and Portfolio Polish

### Person A — CS

- Implement FastAPI endpoints:
  - `GET /health`
  - `POST /transcribe`
- Return transcript, timestamps, selected VAD method, VAD statistics, runtime, ASR RTF, and pipeline RTF.
- Implement file validation and clear errors for unsupported, empty, corrupt, or silence-only audio.
- Write API and evaluation tests.
- Write README sections: architecture, installation, quick start, CLI, API, metrics, results, limitations, future work.

### Person B — ECE

- Finish preprocessing, WebRTC framing, and VAD boundary tests.
- Test sample-rate conversion, stereo-to-mono conversion, PCM frame validity, silence handling, merge logic, padding limits, and invalid audio.
- Finalize DSP methodology, dataset/annotation documentation, and visualizations.
- Record a 30–60 second GIF/video showing CLI or API transcription and VAD diagnostics.
- Test API input files across clean, pause-heavy, noisy, and silence-only cases.

### Shared checkpoint

- Run `pytest` and fix failures.
- Fresh-clone the repository and follow the README from setup to benchmark/API demo.
- Confirm every README table/claim is reproducible.
- Prepare a 45-second explanation per person.
- Final review, final PRs, tag/release, and pin repository on GitHub.

### End-of-day deliverables

- Working CLI and FastAPI service.
- Passing tests.
- Final README, benchmark tables, DSP figures, and error analysis.
- Demo GIF/video.
- CV-ready contribution statements.

---

## 9. CLI and API Contract

### Required CLI

```bash
python -m src.pipeline \
  --audio data/demo/pause_heavy.wav \
  --vad webrtc \
  --aggressiveness 2 \
  --asr-model base
```

Supported VAD options:

```text
--vad none
--vad webrtc
--vad silero
```

### Required API

#### Health check

```http
GET /health
```

Response:

```json
{"status": "ok"}
```

#### Transcribe audio

```http
POST /transcribe
```

Response example:

```json
{
  "file_name": "sample_vi.wav",
  "vad_method": "silero",
  "audio_duration_seconds": 12.4,
  "audio_passed_to_asr_seconds": 8.1,
  "audio_passed_to_asr_ratio": 0.653,
  "asr_runtime_seconds": 2.3,
  "pipeline_runtime_seconds": 2.7,
  "asr_rtf": 0.185,
  "pipeline_rtf": 0.218,
  "segments": [
    {"start_s": 0.31, "end_s": 4.62, "text": "Xin chào, đây là một ví dụ."}
  ],
  "transcript": "Xin chào, đây là một ví dụ."
}
```

---

## 10. Repository Structure

```text
vietspeech-vad-asr/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── README.md
│   ├── test_manifest.csv
│   ├── demo/
│   └── boundaries/
├── src/
│   ├── __init__.py
│   ├── audio.py                 # Person B
│   ├── features.py              # Shared
│   ├── energy_vad.py            # Person A, diagnostic only
│   ├── webrtc_vad.py            # Person B
│   ├── silero_vad.py            # Person A
│   ├── segmenter.py             # Shared
│   ├── asr.py                   # Person A
│   ├── evaluation.py            # Person A
│   ├── pipeline.py              # Person A
│   ├── schemas.py               # Person A
│   └── api.py                   # Person A
├── scripts/
│   ├── benchmark.py             # Person A
│   └── plot_vad_analysis.py     # Person B
├── tests/
│   ├── test_audio.py            # Person B
│   ├── test_webrtc_vad.py       # Person B
│   ├── test_segmenter.py        # Shared
│   ├── test_evaluation.py       # Person A
│   └── test_api.py              # Person A
├── results/
│   ├── benchmark.csv
│   ├── error_analysis.md
│   └── figures/
└── demo/
    └── demo.gif
```

---

## 11. Git and Collaboration Rules

### Branches

```text
main
├── feature/asr-evaluation-api       # Person A
├── feature/audio-webrtc-vad         # Person B
└── feature/silero-vad               # Person A
```

### Rules

- Use small commits with clear messages.
- Open pull requests for meaningful completed units of work.
- At least one teammate reviews every PR before merging to `main`.
- Do not change the frozen evaluation manifest during final benchmark execution.
- Record configuration and machine details with benchmark results.
- Use issues/tasks to track blockers and handoffs.

### Suggested commit messages

```text
feat: add audio inspection and PCM16 conversion
feat: add 20 ms WebRTC VAD frame segmentation
feat: add faster-whisper baseline transcription
feat: add Silero VAD adapter
feat: add common segment postprocessor
feat: add Vietnamese WER CER evaluation
feat: add VAD-gated benchmark runner
feat: add FastAPI transcription endpoint
test: cover silence and segment boundary cases
docs: add benchmark results and DSP analysis
```

---

## 12. Definition of Done

The project is complete only when all conditions below are true:

- [ ] Full-audio, WebRTC-VAD-gated, and Silero-VAD-gated ASR run through one CLI.
- [ ] Input audio is standardized to 16 kHz mono.
- [ ] WebRTC VAD uses valid PCM16 frames and configured aggressiveness.
- [ ] Silero VAD produces compatible timestamp segments.
- [ ] Both VAD systems use identical postprocessing rules.
- [ ] The evaluation set has 18–24 manually transcribed Vietnamese clips.
- [ ] Six to eight clips have manual speech-boundary labels.
- [ ] WER, CER, ASR RTF, pipeline RTF, ASR audio ratio, and segment count are reported.
- [ ] Boundary error, IoU, missed-speech, and false-activation examples are documented.
- [ ] Main benchmark compares full-audio, WebRTC VAD, and Silero VAD fairly.
- [ ] `GET /health` and `POST /transcribe` work.
- [ ] Core tests pass.
- [ ] README has reproducible setup, CLI/API instructions, results, figures, limitations, and future work.
- [ ] A short demo GIF/video exists.
- [ ] Each teammate can accurately explain their own contribution and the final result.

---

## 13. CV and Interview Positioning

### Person A — CV bullets

**VIETSPEECH: VIETNAMESE VAD-GATED ASR EVALUATION AND API PIPELINE**  
*Python, PyTorch, faster-whisper, Silero VAD, WebRTC VAD, FastAPI, JiWER, NumPy*

- Built the ASR, neural-VAD, evaluation, and FastAPI layers of an offline Vietnamese transcription pipeline; compared full-audio, WebRTC-VAD-gated, and Silero-VAD-gated inference using faster-whisper.
- Designed reproducible experiments on a manually transcribed Vietnamese dataset, reporting WER, CER, ASR/pipeline real-time factor, ASR audio ratio, segment counts, and boundary-quality diagnostics.
- Implemented RMS-energy analysis and investigated how frame-level decisions, VAD sensitivity, padding, and missed-speech errors affect Vietnamese transcription quality and compute efficiency.
- Delivered a testable FastAPI service with structured transcript timestamps and VAD/runtime statistics, including tests for audio validation, silence handling, segmentation, and evaluation logic.

### Person B — CV bullets

- Developed the DSP/audio frontend for a Vietnamese ASR benchmark, including 16 kHz mono PCM16 conversion, 20 ms frame processing, RMS/ZCR analysis, WebRTC VAD integration, and speech-segment postprocessing.
- Evaluated WebRTC and neural VAD behavior with manually labelled speech boundaries, waveform/spectrogram diagnostics, boundary error, IoU, missed-speech cases, and false activations.
- Curated and annotated a controlled Vietnamese speech dataset spanning clean, pause-heavy, noisy, short-utterance, and code-switched conditions.

### 45-second team explanation

> We built an offline Vietnamese ASR benchmark that compares transcription of complete audio files with transcription after WebRTC VAD or Silero VAD segmentation. We standardized all recordings to 16 kHz mono, used a shared segment postprocessor, and transcribed detected speech regions with faster-whisper. We measured WER, CER, ASR and end-to-end real-time factor, the percentage of audio sent to ASR, and VAD boundary behavior across clean, pause-heavy, and moderately noisy Vietnamese recordings. The project demonstrates the practical trade-off: VAD can reduce unnecessary ASR processing, but overly aggressive segmentation can clip quiet speech and degrade recognition quality.

---

## 14. Key Reminder

Do not optimize for the number of features. Optimize for a **clean, fair, reproducible experiment** with Vietnamese audio, real metrics, correct engineering decisions, clear ownership, and a polished GitHub presentation.
