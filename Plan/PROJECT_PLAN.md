# VietSpeech: DSP-Aware VAD Benchmark for Vietnamese ASR
# Kế hoạch Dự án / Project Plan (Bilingual: Tiếng Việt & English)

---

# PHẦN I: BẢN KẾ HOẠCH DỰ ÁN (TIẾNG VIỆT)

## 📌 Bối cảnh & Mục tiêu Tổng quan

> **Đề bài:** Tinh chỉnh kế hoạch cho Người A (chuyên môn Khoa học Máy tính - CS) và Người B (chuyên môn Kỹ thuật Điện & Máy tính - ECE). Đảm bảo sau khi hoàn thành dự án, cả hai đều tích lũy thêm kiến thức (ưu tiên về Xử lý Tín hiệu Số - DSP) và dự án có tính khả thi cao để hoàn thành trong đúng 4 ngày.

Đối với một dự án 4 ngày gồm 2 người, hãy giữ phạm vi thực sự tập trung: xây dựng và đánh giá chuẩn (benchmark) một hệ thống nhận dạng giọng nói tiếng Việt (ASR) dạng offline có kiểm soát bởi bộ phát hiện hoạt động giọng nói (VAD-gated) cho các file âm thanh. Bỏ qua streaming micro thời gian thực, giao diện UI, phân tách người nói (diarization), và huấn luyện mô hình tùy chỉnh; hoàn thành với một CLI có thể tái lập, endpoint FastAPI, một bộ dữ liệu gán nhãn nhỏ và một bài phân tích/so sánh trọng tâm về DSP rõ ràng.

### Câu hỏi nghiên cứu trung tâm:
> *"Các bước tiền xử lý âm thanh và thiết lập VAD ảnh hưởng như thế nào đến độ chính xác ASR tiếng Việt, hiệu suất suy luận (inference efficiency) và chất lượng ranh giới đoạn nói (speech-boundary quality)?"*

Sử dụng **WebRTC VAD** làm baseline định hướng DSP và **Silero VAD** làm đối trọng so sánh dựa trên Neural VAD (Deep Learning), sau đó phiên âm các đoạn giọng nói bằng **faster-whisper**. WebRTC VAD đặc biệt hữu ích cho việc học tập vì nó hoạt động trực tiếp trên các khung (frames) âm thanh PCM rõ ràng — 16-bit mono, lấy mẫu ở 8/16/32/48 kHz, trong các khung 10/20/30 ms — giúp bạn làm việc trực tiếp với các khái niệm cốt lõi đằng sau xử lý giọng nói số theo thời gian thực. Silero VAD mang lại một sự so sánh với mô hình học sâu và hỗ trợ âm thanh 8 kHz và 16 kHz. Faster-whisper là bản triển khai Whisper dựa trên CTranslate2 được thiết kế để tối ưu hóa hiệu năng suy luận.

---

## 🏷️ Phạm vi Dự án Cuối cùng

* **Tên dự án:** `VietSpeech: DSP-Aware VAD Benchmark for Vietnamese ASR`
* **Sơ đồ Pipeline Hệ thống (System Pipeline):**

```text
File âm thanh tiếng Việt (Vietnamese audio file)
        │
        ▼
Tiền xử lý (Preprocessing)
[Giải mã ➔ Mono ➔ 16 kHz ➔ Tín hiệu dạng sóng PCM / Float]
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
   WebRTC VAD                              Silero VAD
(frame-based DSP baseline)          (neural VAD comparison)
        │                                      │
        └──────── speech timestamp segments ───┘
                           │
                           ▼
              Segment padding and merging
              (Đệm và gộp các phân đoạn)
                           │
                           ▼
             faster-whisper Vietnamese ASR
                           │
                           ▼
       Transcript + WER/CER + RTF + VAD metrics
```

---

## 📦 Các Sản phẩm Bàn giao Bắt buộc (Required Deliverables)

1. **Một Python CLI có thể tái lập:**
   ```bash
   python -m src.pipeline \
     --audio data/demo/pause_heavy.wav \
     --vad webrtc \
     --aggressiveness 2 \
     --asr-model base
   ```
2. **Hai phương pháp VAD:**
   - **WebRTC VAD:** Baseline chuẩn DSP / hệ thống.
   - **Silero VAD:** Đối trọng so sánh Neural VAD.
3. **Một ASR backend:**
   - `faster-whisper`, cấu hình `language="vi"`.
4. **Một bộ dữ liệu đánh giá gán nhãn thủ công (Evaluation set):**
   - Tổng cộng 16–24 file ghi âm tiếng Việt.
   - Bao gồm: Giọng nói rõ ràng (Clean), nhiều khoảng dừng (Pause-heavy), tiếng ồn vừa phải (Moderate-noise), câu ngắn (Short utterances), và câu kỹ thuật/pha tiếng Anh (Technical/code-switched).
   - 1 bản transcript chuẩn (Reference transcript) cho mỗi bản ghi.
   - *(Tùy chọn nhưng rất giá trị):* Đánh dấu thủ công mốc thời gian bắt đầu/kết thúc đoạn nói cho 6–8 file để kiểm tra hành vi ranh giới VAD.
5. **Một bộ benchmark so sánh:**
   - ASR toàn bộ audio (Whole-audio baseline - Không qua VAD).
   - WebRTC-VAD-gated ASR (ASR qua bộ lọc WebRTC VAD).
   - Silero-VAD-gated ASR (ASR qua bộ lọc Silero VAD).
6. **Các chỉ số đo lường (Metrics):**
   - WER (Word Error Rate) và CER (Character Error Rate).
   - Real-Time Factor (RTF).
   - Tỷ lệ thời lượng giọng nói giữ lại / Tổng thời lượng audio (Speech duration retained / audio duration).
   - Số lượng đoạn VAD (Number of VAD segments).
   - Các ví dụ kích hoạt sai (False activation) và bỏ sót giọng nói (Missed-speech).
   - Sai số ranh giới tùy chọn cho tập 6–8 file có phân đoạn thủ công.
7. **Một FastAPI endpoint cơ bản:**
   - `GET /health`
   - `POST /transcribe`
8. **README, bộ unit tests, bảng kết quả, sơ đồ kiến trúc và một video/GIF demo ngắn.**

---

## 👥 Phân chia Vai trò Nhóm (Team Roles)

Sự phân chia này cho phép cả hai thành viên cùng đóng góp vào hệ thống tích hợp hoàn chỉnh trong khi phát triển các kỹ năng riêng biệt. **Người A** thu nhận kiến thức thực tế về DSP/phân tích âm thanh; **Người B** tích lũy kiến thức phần mềm định hướng production và đánh giá hệ thống, đồng thời tự tay lập trình xử lý giọng nói ở cấp độ khung (frame-level).

| Hạng mục công việc (Workstream) | Người A — Khoa học Máy tính (CS) | Người B — Kỹ thuật Điện & Máy tính (ECE) |
| :--- | :--- | :--- |
| **Phụ trách chính** | Kiến trúc hệ thống, tích hợp ASR, đánh giá (Evaluation) và tích hợp API. | Tiền xử lý DSP/Audio, WebRTC VAD, phân tích và kiểm chứng tín hiệu. |
| **Phát triển kỹ năng chính** | Đặc trưng âm thanh, Spectrogram, điểm ngắt VAD, phương pháp luận thực nghiệm. | Pipeline Speech-AI, thiết kế dịch vụ Python, đo lường và triển khai ASR. |
| **Tiền xử lý âm thanh** | Tích hợp interface chung và xử lý file I/O. | Lập trình chuyển đổi PCM, chia khung (framing), tính RMS/năng lượng, kiểm tra resampling. |
| **VAD** | Tích hợp Silero VAD và xử lý hậu kỳ phân đoạn linh hoạt. | Viết wrapper cho WebRTC VAD, logic xử lý khung, thực nghiệm các mức độ hung hăng (aggressiveness). |
| **ASR** | Lập trình nhận dạng bằng faster-whisper và khôi phục mốc thời gian (timestamp). | Kiểm thử đầu vào ASR và phân tích ảnh hưởng của việc cắt đoạn tới kết quả ASR. |
| **Đánh giá (Evaluation)** | Lập trình tính WER, CER, RTF, báo cáo benchmark tổng hợp. | Tạo chẩn đoán VAD: quyết định trên từng frame, tỷ lệ giữ giọng nói, các trường hợp ranh giới. |
| **Bộ dữ liệu** | Tạo manifest thực nghiệm và quy tắc chuẩn hóa văn bản. | Thu âm/thu thập dữ liệu âm thanh, chép transcript thủ công, gán nhãn ranh giới 6–8 clip. |
| **API** | Xây dựng FastAPI endpoint và schema phản hồi. | Kiểm tra các trường hợp biên khi upload file, tạo audio test API. |
| **Kiểm thử (Tests)** | Viết tests cho pipeline, evaluation và API. | Viết tests cho DSP/tiền xử lý/VAD frame và boundary. |
| **Tài liệu** | Kiến trúc, cách cài đặt, kết quả, hạn chế dự án. | Phần giải thích DSP, giao thức dữ liệu, biểu đồ minh họa VAD trực quan. |

### Trách nhiệm chung:
- Review Pull Request của nhau thay vì merge code chưa được kiểm tra.
- Pair-program (lập trình cặp) phần tích hợp mỗi ngày 1 lần.
- Cùng chạy một benchmark trước khi chốt kết quả cuối cùng.
- Cùng viết README và chuẩn bị bản demo.
- Cả hai người đều phải có khả năng giải thích cả 2 phương pháp VAD, phương pháp ASR và kết quả thực nghiệm cuối cùng.

---

## 🌿 Quy trình Git & Pull Requests

```text
main
├── feature/asr-evaluation-api     # Người A
├── feature/audio-webrtc-vad       # Người B
└── feature/silero-vad             # Người A, sau khi interface tiền xử lý hoàn tất
```

**Sử dụng các pull request nhỏ:**
- `feat: audio loading and PCM validation`
- `feat: WebRTC VAD frame segmentation`
- `feat: faster-whisper baseline transcription`
- `feat: Silero VAD adapter`
- `feat: WER CER benchmark`
- `feat: FastAPI transcription endpoint`
- `test: silence and VAD-boundary cases`

---

## 🔬 Kiến thức DSP Tích hợp (DSP Learning Built-in)

Dự án chủ đích đưa vào một phần phân tích DSP trực quan, thay vì coi VAD như một thư viện hộp đen.

### 🔹 Người B: Phần việc DSP Bắt buộc
Lập trình và giải thích các yếu tố sau:

| Khái niệm DSP | Nhiệm vụ thực hành trong dự án |
| :--- | :--- |
| **Tần số lấy mẫu (Sampling rate)** | Chuyển đổi toàn bộ âm thanh đầu vào về 16 kHz và giải thích rằng tần số Nyquist là 8 kHz, bao phủ dải tần giọng nói chính được sử dụng bởi nhiều hệ thống nhận dạng tiếng nói. |
| **Lượng tử hóa / PCM** | Chuyển đổi âm thanh sang signed 16-bit mono PCM cho WebRTC VAD. |
| **Chia khung (Framing)** | Chia âm thanh thành các khung 20 ms; ở 16 kHz, mỗi khung chứa chính xác: $$N = f_s \times T = 16000 \times 0.020 = 320\text{ mẫu (samples)}$$ *(Tương thích trực tiếp với các thời lượng khung hợp lệ của WebRTC VAD).* |
| **Phân tích ngắn hạn (Short-time analysis)** | Tính năng lượng RMS trên từng khung và tỷ lệ đổi dấu (Zero-Crossing Rate - ZCR). |
| **Phổ Spectrogram** | Vẽ log-mel spectrogram hoặc độ lớn STFT kết hợp vẽ đè các vùng giọng nói dự đoán của VAD. |
| **Đánh đổi ngưỡng (Thresholding trade-off)** | So sánh các chế độ Aggressiveness 1, 2 và 3 của WebRTC VAD. |
| **Logic ghép đoạn (Segmentation logic)** | Gộp các khung có tiếng gần nhau và thêm đệm (padding) để tránh cắt cụt phụ âm đầu/cuối. |

### 🔹 Người A: Phần việc DSP Bắt buộc
Người A không nên chỉ xử lý code ứng dụng thuần túy. Bổ sung các nhiệm vụ sau:
- Lập trình một baseline VAD dựa trên ngưỡng năng lượng RMS cấp độ khung (Frame-level RMS-energy baseline VAD).
- So sánh Energy VAD với WebRTC VAD trên 6–8 file có gán nhãn ranh giới.
- Vẽ biểu đồ năng lượng theo thời gian cùng với quyết định speech / non-speech.
- Giải thích tại sao một ngưỡng năng lượng đơn giản có thể thất bại:
  - Tiếng ồn nền làm tăng mức năng lượng tổng thể.
  - Giọng nói nhỏ/thì thầm có năng lượng thấp.
  - Tiếng gõ bàn phím hoặc tiếng đóng cửa tạo ra năng lượng cao gây kích hoạt sai (False positives).
- So sánh định tính giữa Energy VAD, WebRTC VAD và Silero VAD, ngay cả khi chỉ có WebRTC và Silero được đưa vào benchmark ASR cuối cùng.
- Đảm nhiệm phần phương pháp luận: giải thích tại sao tiền xử lý, kích thước khung, padding và phân đoạn VAD có thể ảnh hưởng đến độ chính xác và runtime của ASR.

**Ý nghĩa:**
- **Người B** học về chuẩn bị tín hiệu thực tế và VAD dạng khung.
- **Người A** học về phân tích giọng nói ngắn hạn và cách các lựa chọn DSP ảnh hưởng đến một hệ thống ML/ASR.

---

## 📅 Kế hoạch Chi tiết 4 Ngày (Four-Day Plan)

### 📍 Ngày 1 — Baseline, Dữ liệu và Định dạng Âm thanh
*Mục tiêu: Nhận dạng hoàn chỉnh 1 file tiếng Việt từ đầu đến cuối và thiết lập định dạng âm thanh chuẩn hóa.*

| Người A — CS | Người B — ECE | Checkpoint chung |
| :--- | :--- | :--- |
| - Tạo cấu trúc repo, môi trường Python, `requirements.txt`, và script baseline faster-whisper.<br>- Lập trình xuất JSON cho ASR toàn bộ file đơn giản và bộ đếm thời gian.<br>- Bắt đầu viết các hàm chuẩn hóa văn bản tính WER/CER. | - Lập trình hàm kiểm tra âm thanh: thời lượng, sample rate, channels, dtype; viết hàm chuyển đổi sang 16 kHz mono PCM WAV.<br>- Viết hàm chia khung 20 ms và tính năng lượng RMS từng khung.<br>- Thu âm/thu thập và chép transcript thủ công 8–10 clip; tạo file `test_manifest.csv` ban đầu. | - Chạy baseline ASR trên 3–5 file tiếng Việt.<br>- Thống nhất quy ước gán nhãn tập dữ liệu.<br>- Commit baseline và định dạng data-manifest. |

**Sản phẩm bàn giao cuối Ngày 1:**
- Baseline ASR hoạt động được.
- Mọi file đầu vào đều có thể chuyển đổi về 16 kHz mono.
- 8–10 clip có transcript chuẩn đã được xác thực.
- Có biểu đồ RMS-energy cho ít nhất 1 bản ghi sạch và 1 bản ghi nhiễu.

---

### 📍 Ngày 2 — Xây dựng VAD Baselines và Chẩn đoán DSP
*Mục tiêu: Xây dựng 3 góc nhìn VAD: Ngưỡng năng lượng, WebRTC VAD và Silero VAD.*

| Người A — CS | Người B — ECE | Checkpoint chung |
| :--- | :--- | :--- |
| - Lập trình RMS-energy VAD; thêm ngưỡng năng lượng có thể cấu hình.<br>- Tích hợp Silero VAD và trả về timestamp.<br>- Viết script so sánh VAD cơ bản. | - Lập trình WebRTC VAD với khung 20 ms, 16-bit PCM và các mode 1/2/3.<br>- Thêm logic chuyển frame sang segment, gộp đoạn, padding và chống cắt viền.<br>- Gán nhãn thủ công khoảng giọng nói cho 6–8 clip và xác định false positives / false negatives. | - Thống nhất định dạng JSON phân đoạn VAD dùng chung.<br>- Đánh giá lại các vùng giọng nói bằng cách nghe và xem biểu đồ.<br>- Chọn thiết lập mặc định để test vào Ngày 3. |

**Sản phẩm bàn giao cuối Ngày 2:**
- Mọi file đầu vào đều có thể tạo ra: `energy-VAD timestamps`, `WebRTC-VAD timestamps`, `Silero-VAD timestamps`.
- Một biểu đồ trực quan đè dạng sóng (hoặc năng lượng RMS) kết hợp các đoạn VAD.
- Nhóm đã tài liệu hóa các tham số phân đoạn mặc định.

---

### 📍 Ngày 3 — Ghép nối VAD-gated ASR và Chạy Benchmark
*Mục tiêu: Hoàn thành thí nghiệm cốt lõi của dự án.*

| Người A — CS | Người B — ECE | Checkpoint chung |
| :--- | :--- | :--- |
| - Xây dựng pipeline đưa phân đoạn VAD vào ASR cho: Full audio, WebRTC VAD, và Silero VAD.<br>- Lập trình tính WER, CER, RTF, runtime, ghép transcript, xuất file JSON/SRT.<br>- Tạo file CSV benchmark và thống kê tóm tắt. | - Mở rộng tập test lên 16–24 clip bao gồm: sạch, nhiều khoảng dừng, nhiễu vừa, câu ngắn, và câu pha tiếng Anh.<br>- Tính toán các chẩn đoán VAD: tỷ lệ giọng nói, số đoạn, ví dụ viền, ghi chú kích hoạt sai / bỏ sót tiếng.<br>- Tạo biểu đồ spectrogram / RMS / VAD cho 2 trường hợp âm thanh đại diện. | - Chạy cả 3 hệ thống trên cùng một file manifest chuẩn cố định.<br>- Cùng nhau phân tích 5 ví dụ có kết quả kém nhất.<br>- Thống nhất bảng kết quả và kết luận cốt lõi. |

**Sản phẩm bàn giao cuối Ngày 3:**
- Benchmark so sánh:
  1. Whole-audio ASR
  2. WebRTC-VAD-gated ASR
  3. Silero-VAD-gated ASR
- Ít nhất 1 bảng kết quả hoàn chỉnh và 2 biểu đồ chẩn đoán.
- Ghi chú phân tích lỗi cho 5 trường hợp lỗi tiêu biểu và cung cấp nhiều thông tin nhất.

---

### 📍 Ngày 4 — API, Kiểm thử, Kết quả và Hoàn thiện
*Mục tiêu: Biến thí nghiệm đang hoạt động thành một dự án portfolio hoàn chỉnh.*

| Người A — CS | Người B — ECE | Checkpoint chung |
| :--- | :--- | :--- |
| - Lập trình FastAPI `GET /health` và `POST /transcribe`; trả về transcript, timestamps, thống kê VAD và runtime.<br>- Viết unit tests bằng `pytest` cho module evaluation và phản hồi API.<br>- Viết README: quick start, quy trình benchmark, ví dụ gọi API, và các hạn chế. | - Thêm unit tests cho tiền xử lý / WebRTC: sample rate, stereo-to-mono, đầu vào câm (silence), độ dài khung hợp lệ, ranh giới gộp/padding.<br>- Hoàn thiện phần giải thích DSP, hình vẽ minh họa và tài liệu hóa dữ liệu âm thanh.<br>- Tạo GIF/video demo và chú thích trực quan hóa VAD. | - Test API với 4 loại file: sạch, nhiều khoảng dừng, nhiễu, và file hoàn toàn im lặng.<br>- Test clone repo và cài đặt lại từ đầu theo README.<br>- Mỗi người chuẩn bị bài thuyết trình 45 giây giải thích về phần việc của mình. |

**Sản phẩm bàn giao cuối Ngày 4:**
- CLI và FastAPI endpoint hoạt động hoàn chỉnh.
- Toàn bộ unit tests passed.
- README đầy đủ: kiến trúc, phương pháp, kết quả, phân tích DSP, hạn chế, và câu lệnh tái lập kết quả.
- Demo GIF / video.
- Các gạch đầu dòng mô tả đóng góp sẵn sàng đưa vào CV.

---

## 🚫 Tránh Mở rộng Phạm vi Dư thừa (Avoid Scope Creep)

Những phần sau **nằm ngoài phạm vi 4 ngày một cách rõ ràng**:
- ❌ Streaming micro thời gian thực.
- ❌ Tự huấn luyện (custom training) hoặc fine-tuning mô hình ASR.
- ❌ Phân tách người nói (Speaker diarization).
- ❌ Giao diện frontend Streamlit phức tạp.
- ❌ Triển khai Docker, trừ khi toàn bộ tính năng cốt lõi và tài liệu đã hoàn thành sớm.
- ❌ Nhiều hơn 2 phương pháp VAD trong bảng benchmark ASR chính.
- ❌ Một tập dữ liệu công cộng quá lớn hoặc hàng trăm mẫu gán nhãn thủ công.

*Lưu ý: Energy VAD là một baseline phục vụ học tập DSP, không nhất thiết là điều kiện pipeline thứ ba trong benchmark ASR cuối cùng. Bảng kết quả cuối cùng nên giữ sự tập trung: Không dùng VAD, WebRTC VAD, và Silero VAD.*

---

## 📊 Bảng Kết quả Thực nghiệm Cuối cùng (Final Experiment Table)

Sử dụng mẫu này và chỉ điền vào với các số liệu đo đạc thực tế:

| Pipeline | Điều kiện âm thanh (Audio condition) | WER (%) ↓ | CER (%) ↓ | RTF ↓ | Tỷ lệ giữ tiếng (Speech retained) | Số đoạn VAD/file | Nhận xét chính (Key observation) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full-audio ASR** | Clean (Sạch) | — | — | — | 100% | N/A | Baseline |
| **WebRTC VAD + ASR** | Clean | — | — | — | —% | — | — |
| **Silero VAD + ASR** | Clean | — | — | — | —% | — | — |
| **Full-audio ASR** | Pause-heavy (Nhiều ngắt quãng) | — | — | — | 100% | N/A | Baseline |
| **WebRTC VAD + ASR** | Pause-heavy | — | — | — | —% | — | — |
| **Silero VAD + ASR** | Pause-heavy | — | — | — | —% | — | — |
| **Full-audio ASR** | Moderate noise (Nhiễu vừa) | — | — | — | 100% | N/A | Baseline |
| **WebRTC VAD + ASR** | Moderate noise | — | — | — | —% | — | — |
| **Silero VAD + ASR** | Moderate noise | — | — | — | —% | — | — |

### Bảng hàng giá trị trung bình tổng hợp:
| Pipeline | Mean WER ↓ | Mean CER ↓ | Mean RTF ↓ | Mean speech retained ↓ | Boundary behavior (Hành vi ranh giới) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full-audio ASR** | — | — | — | 100% | N/A |
| **WebRTC VAD + ASR** | — | — | — | —% | — |
| **Silero VAD + ASR** | — | — | — | —% | — |

---

## 🛠️ Stack Triển khai Tối giản (Minimal Implementation Stack)

```text
Python 3.10+
PyTorch + torchaudio       # Thao tác dạng sóng, resampling
soundfile                  # Đọc/ghi file WAV
webrtcvad-wheels           # WebRTC VAD
silero-vad                 # Neural VAD
faster-whisper             # ASR engine tối ưu qua CTranslate2
jiwer                      # Tính toán WER / CER
numpy + pandas             # Xử lý dữ liệu và bảng kết quả
matplotlib                 # Vẽ waveform, RMS, spectrogram plots
fastapi + uvicorn          # REST API
pytest + httpx             # Bộ kiểm thử tests
pyyaml                     # Quản lý cấu hình thực nghiệm
```

*Sử dụng `webrtcvad-wheels` thay vì gói cũ vốn khó khăn trong tương thích cài đặt. Nếu việc cài đặt WebRTC làm nghẽn tiến độ, hãy ghi nhận nó như một vấn đề thiết lập môi trường, sử dụng pre-packaged wheel, và đưa Silero thành luồng VAD chính; không hy sinh sản phẩm bàn giao cuối cùng chỉ để giải quyết vấn đề phụ thuộc.*

---

## 📂 Cấu trúc Thư mục Repository Đề xuất

```text
vietspeech-dsp-vad-asr/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── README.md
│   └── test_manifest.csv
├── src/
│   ├── audio.py              # Người B
│   ├── features.py           # Dùng chung: helper tính RMS, STFT
│   ├── energy_vad.py         # Người A
│   ├── webrtc_vad.py         # Người B
│   ├── silero_vad.py         # Người A
│   ├── segmenter.py          # Dùng chung
│   ├── asr.py                # Người A
│   ├── evaluation.py         # Người A
│   ├── pipeline.py           # Người A
│   └── api.py                # Người A
├── scripts/
│   ├── benchmark.py          # Người A
│   └── plot_vad_analysis.py  # Người B
├── tests/
│   ├── test_audio.py         # Người B
│   ├── test_webrtc_vad.py    # Người B
│   ├── test_segmenter.py     # Dùng chung
│   ├── test_evaluation.py    # Người A
│   └── test_api.py           # Người A
├── results/
│   ├── benchmark.csv
│   ├── error_analysis.md
│   └── figures/
└── demo/
    └── demo.gif
```

---

## 💼 Mô tả Đóng góp Sẵn sàng cho CV & Trả lời Phỏng vấn

### 📝 Gạch đầu dòng cho CV (CV-ready contributions):

- **Người A — Computer Science:**
  > • *Built the ASR and evaluation layer of a Vietnamese VAD-gated transcription system using faster-whisper, Silero VAD, FastAPI, and JiWER; benchmarked full-audio, WebRTC-VAD, and neural-VAD pipelines using WER, CER, real-time factor, and segment-level error analysis.*  
  > • *Implemented an RMS-energy VAD baseline and analyzed how framing, energy thresholds, segment padding, and VAD false negatives affected Vietnamese ASR quality and runtime.*

- **Người B — Electrical & Computer Engineering:**
  > • *Developed the DSP/audio front end for a Vietnamese ASR pipeline, including 16 kHz mono PCM conversion, 20 ms frame processing, RMS-energy analysis, WebRTC VAD integration, and speech-segment postprocessing.*  
  > • *Compared frame-based VAD aggressiveness settings using waveform/spectrogram diagnostics and annotated boundary cases, identifying trade-offs among noise rejection, speech clipping, and ASR compute reduction.*

### 🎤 Đoạn giải thích 45 giây dùng trong Phỏng vấn (Interview Explanation):
> *"We built an offline Vietnamese ASR benchmark that compares transcription of the full audio with transcription after WebRTC VAD or Silero VAD segmentation. We standardized recordings to 16 kHz mono, used frame-level analysis to inspect energy and speech boundaries, and sent detected speech regions to faster-whisper. We measured WER, CER, real-time factor, speech retained, and segmentation failures across clean, pause-heavy, and moderately noisy Vietnamese recordings. The comparison let us study the practical trade-off: aggressive VAD can reduce unnecessary ASR processing, but it can also clip quiet speech or word boundaries."*

---

## 📚 Tài liệu Tham khảo (References)
1. https://github.com/wiseman/py-webrtcvad/blob/master/README.rst
2. https://github.com/snakers4/silero-vad
3. https://github.com/SYSTRAN/faster-whisper
4. https://github.com/snakers4/silero-vad/issues/2
5. https://runthisai.com/en/blog/faster-whisper-guide
6. https://raw.githubusercontent.com/NeverSight/skills_feed/refs/heads/main/data/skills-md/theplasmak/faster-whisper/faster-whisper/SKILL.md
7. https://runthisai.com/en/tool/faster-whisper
8. https://doc.rapida.ai/opensource/services/assistant-api/vad/silero
9. https://openapps.pro/packages/faster-whisper
10. https://github.com/AIXerum/faster-whisper
11. https://github.com/bungerr/faster-whisper-3
12. https://github.com/deeenancy/faster-whisper-CT2
13. https://github.com/openai/whisper/discussions/937
14. https://github.com/BBC-Esq/Faster-Whisper-Transcriber
15. https://github.com/snakers4/silero-vad/releases

---
---

# PART II: COMPLETE PROJECT PLAN (ENGLISH)

## 📌 Context & Objectives

> **Task Prompt:** Refine the plan for person A with expertise in computer science and person B in electrical and computer engineering. Make sure that after completing the project, both can gain more knowledge (preferably in DSP) and the project can be practically completed in 4 days.

For a 4-day, two-person project, keep the scope focused: build and benchmark an offline Vietnamese VAD-gated ASR system for audio files. Skip live microphone streaming, a UI, diarization, and custom model training; finish with a reproducible CLI, FastAPI endpoint, a small labeled dataset, and one clear DSP-centered comparison.

### Central question:
> *"How do audio preprocessing and VAD settings affect Vietnamese ASR accuracy, inference efficiency, and speech-boundary quality?"*

Use **WebRTC VAD** as the DSP-oriented baseline and **Silero VAD** as the neural VAD comparison, then transcribe their speech segments using **faster-whisper**. WebRTC VAD is particularly useful for learning because it works on explicitly framed PCM audio—16-bit mono, sampled at 8/16/32/48 kHz, in 10/20/30 ms frames—so you will directly work with the concepts behind real-time digital speech processing. Silero VAD provides a learned VAD comparison and supports 8 kHz and 16 kHz audio. Faster-whisper is a CTranslate2-based implementation of Whisper designed for efficient inference.

---

## 🏷️ Final Project Scope

* **Project title:** `VietSpeech: DSP-Aware VAD Benchmark for Vietnamese ASR`
* **System pipeline:**

```text
Vietnamese audio file
        │
        ▼
Preprocessing
Decode ➔ mono ➔ 16 kHz ➔ PCM / float waveform
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
   WebRTC VAD                              Silero VAD
(frame-based DSP baseline)          (neural VAD comparison)
        │                                      │
        └──────── speech timestamp segments ───┘
                           │
                           ▼
              Segment padding and merging
                           │
                           ▼
             faster-whisper Vietnamese ASR
                           │
                           ▼
       Transcript + WER/CER + RTF + VAD metrics
```

---

## 📦 Required Deliverables

1. **A reproducible Python CLI:**
   ```bash
   python -m src.pipeline \
     --audio data/demo/pause_heavy.wav \
     --vad webrtc \
     --aggressiveness 2 \
     --asr-model base
   ```
2. **Two VAD methods:**
   - **WebRTC VAD:** DSP/system-style baseline.
   - **Silero VAD:** neural VAD comparison.
3. **One ASR backend:**
   - `faster-whisper`, configured with `language="vi"`.
4. **A manually labeled evaluation set:**
   - 16–24 Vietnamese recordings total.
   - Clean speech, pause-heavy speech, moderate-noise speech, short utterances, and technical/code-switched sentences.
   - One reference transcript per recording.
   - *(Optional, but valuable):* manually mark speech start/end intervals for 6–8 files to inspect VAD boundary behavior.
5. **One benchmark:**
   - Whole-audio ASR baseline.
   - WebRTC-VAD-gated ASR.
   - Silero-VAD-gated ASR.
6. **Metrics:**
   - WER and CER.
   - Real-Time Factor (RTF).
   - Speech duration retained / audio duration.
   - Number of VAD segments.
   - False activation and missed-speech examples.
   - Optional boundary error for the small manually segmented subset.
7. **A basic FastAPI endpoint:**
   - `GET /health`
   - `POST /transcribe`
8. **README, tests, results table, architecture diagram, and a short demo video/GIF.**

---

## 👥 Team Roles

This division lets both people contribute to the final integrated system while developing distinct skills. **Person A** gains practical DSP/audio-analysis knowledge; **Person B** gains production-oriented software and evaluation knowledge while also implementing frame-level speech processing.

| Workstream | Person A — Computer Science | Person B — Electrical & Computer Engineering |
| :--- | :--- | :--- |
| **Main ownership** | System architecture, ASR integration, evaluation and API integration. | DSP/audio preprocessing, WebRTC VAD, signal analysis and validation. |
| **Primary technical growth** | Audio features, spectrograms, VAD endpointing, experimental methodology. | Speech-AI pipelines, Python service design, ASR metrics and deployment. |
| **Audio preprocessing** | Integrate common interface and file handling. | Implement PCM conversion, framing, RMS/energy, resampling checks. |
| **VAD** | Integrate Silero VAD and configurable segment postprocessing. | Implement WebRTC VAD wrapper, frame logic, aggressiveness experiments. |
| **ASR** | Implement faster-whisper transcription and timestamp restoration. | Test ASR inputs and analyze segmentation impact on ASR output. |
| **Evaluation** | Implement WER, CER, RTF, aggregate benchmark reports. | Create VAD diagnostics: frame decisions, retained speech ratio, boundary cases. |
| **Dataset** | Create experiment manifest and normalization rules. | Record/curate audio, manually transcribe, annotate 6–8 speech boundaries. |
| **API** | Implement FastAPI endpoint and response schema. | Validate audio-upload edge cases and create API test audio. |
| **Tests** | Pipeline, evaluation, API tests. | DSP/preprocessing/VAD frame and boundary tests. |
| **Documentation** | Architecture, setup, results, limitations. | DSP section, data protocol, VAD visual examples. |

### Shared Responsibilities:
- Review pull requests rather than merging unreviewed code.
- Pair-program the integration once daily.
- Run the same benchmark before finalizing results.
- Co-author the README and prepare the demo.
- Be able to explain both VAD methods, the ASR method, and the final experimental results.

---

## 🌿 Git Workflow & Small Pull Requests

```text
main
├── feature/asr-evaluation-api     # Person A
├── feature/audio-webrtc-vad       # Person B
└── feature/silero-vad             # Person A, after preprocessing interface is ready
```

**Use small pull requests:**
- `feat: audio loading and PCM validation`
- `feat: WebRTC VAD frame segmentation`
- `feat: faster-whisper baseline transcription`
- `feat: Silero VAD adapter`
- `feat: WER CER benchmark`
- `feat: FastAPI transcription endpoint`
- `test: silence and VAD-boundary cases`

---

## 🔬 DSP Learning Built In

The project should deliberately include a small, visible DSP analysis component, rather than treating VAD as a black-box library.

### 🔹 Person B: Required DSP Work
Implement and explain these elements:

| DSP Concept | Practical Project Task |
| :--- | :--- |
| **Sampling rate** | Convert all inputs to 16 kHz and explain that it has a Nyquist frequency of 8 kHz, covering the main speech band used by many speech systems. |
| **Quantization / PCM** | Convert audio to signed 16-bit mono PCM for WebRTC VAD. |
| **Framing** | Split audio into 20 ms frames; at 16 kHz, each frame contains: $$N = f_s \times T = 16000 \times 0.020 = 320\text{ samples}$$ *(That is directly compatible with WebRTC VAD’s permitted frame durations).* |
| **Short-time analysis** | Compute frame RMS energy and zero-crossing rate. |
| **Spectrogram** | Plot log-mel spectrogram or STFT magnitude with predicted VAD regions overlaid. |
| **Thresholding trade-off** | Compare WebRTC VAD aggressiveness modes 1, 2, and 3. |
| **Segmentation logic** | Merge nearby voiced frames and apply padding to prevent clipped initial/final consonants. |

### 🔹 Person A: Required DSP Learning
Person A should not only handle application code. Add these tasks:
- Implement a frame-level RMS-energy baseline VAD.
- Compare it against WebRTC VAD on 6–8 annotated files.
- Plot energy over time alongside speech/non-speech decisions.
- Explain why a simple energy threshold can fail:
  - Background noise elevates energy.
  - Quiet speech has low energy.
  - Keyboard clicks or door sounds can create high-energy false positives.
- Compare energy VAD, WebRTC VAD, and Silero VAD qualitatively, even if only WebRTC and Silero feed the final ASR benchmark.
- Own the methodology section: explain why preprocessing, frame size, padding, and VAD segmentation can influence ASR accuracy and runtime.

**This makes both roles relevant to DSP:**
- **Person B** learns practical signal preparation and frame-based VAD.
- **Person A** learns short-time speech analysis and how DSP choices influence an ML/ASR system.

---

## 📅 Four-Day Plan

### 📍 Day 1 — Baseline, Data, and Audio Format
*Goal: Get a Vietnamese file transcribed end-to-end and establish a consistent audio format.*

| Person A — CS | Person B — ECE | Shared Checkpoint |
| :--- | :--- | :--- |
| - Create repo structure, Python environment, `requirements.txt`, and baseline faster-whisper script.<br>- Implement simple whole-file ASR output JSON and timer.<br>- Start WER/CER normalization functions. | - Implement audio inspection: duration, sample rate, channels, dtype; implement conversion to 16 kHz mono PCM WAV.<br>- Write a framing utility for 20 ms frames and calculate RMS energy per frame.<br>- Record/collect and manually transcribe 8–10 clips; create initial `test_manifest.csv`. | - Run baseline ASR on 3–5 Vietnamese files.<br>- Agree dataset labeling conventions.<br>- Commit baseline and data-manifest format. |

**End-of-day Deliverable:**
- Baseline ASR works.
- All input can be converted to 16 kHz mono.
- 8–10 clips have verified reference transcripts.
- You have an RMS-energy plot for at least one clean and one noisy recording.

---

### 📍 Day 2 — VAD Baselines and DSP Diagnostics
*Goal: Build three VAD views: energy threshold, WebRTC VAD, and Silero VAD.*

| Person A — CS | Person B — ECE | Shared Checkpoint |
| :--- | :--- | :--- |
| - Implement RMS-energy VAD; add configurable energy threshold.<br>- Integrate Silero VAD and return timestamps.<br>- Create a basic VAD comparison script. | - Implement WebRTC VAD with 20 ms, 16-bit PCM frames and modes 1/2/3.<br>- Add frame-to-segment conversion, segment merging, padding, and boundary clipping protections.<br>- Annotate speech intervals for 6–8 clips and identify false positives/false negatives. | - Agree on a shared VAD-segment JSON format.<br>- Review speech regions by listening and viewing plots.<br>- Choose default settings to test on Day 3. |

**End-of-day Deliverable:**
- Every input file can generate:
  - `energy-VAD timestamps`,
  - `WebRTC-VAD timestamps`,
  - `Silero-VAD timestamps`.
- One visualization overlays waveform or RMS energy plus VAD segments.
- The team has documented default segmentation parameters.

---

### 📍 Day 3 — VAD-gated ASR and Benchmark
*Goal: Complete the core experiment.*

| Person A — CS | Person B — ECE | Shared Checkpoint |
| :--- | :--- | :--- |
| - Build VAD-segment-to-ASR pipeline for full audio, WebRTC VAD, and Silero VAD.<br>- Implement WER, CER, RTF, runtime, transcript merge, JSON/SRT export.<br>- Generate benchmark CSV and summary statistics. | - Expand the evaluation set to 16–24 clips across clean, pause-heavy, moderate noise, short, and technical/code-switched speech.<br>- Compute VAD diagnostics: speech ratio, number of segments, boundary examples, false activations/missed speech notes.<br>- Generate spectrogram/RMS/VAD visualizations for two representative audio cases. | - Run all three systems on the same frozen manifest.<br>- Inspect worst 5 examples together.<br>- Decide result table and key conclusion. |

**End-of-day Deliverable:**
- Benchmark comparing:
  1. Whole-audio ASR
  2. WebRTC-VAD-gated ASR
  3. Silero-VAD-gated ASR
- At least one complete result table and two diagnostic figures.
- Error-analysis notes for the five most informative failures.

---

### 📍 Day 4 — API, Tests, Results, and Polish
*Goal: Turn the working experiment into a portfolio project.*

| Person A — CS | Person B — ECE | Shared Checkpoint |
| :--- | :--- | :--- |
| - Implement FastAPI `GET /health` and `POST /transcribe`; return transcript, timestamps, VAD statistics, and runtime.<br>- Add `pytest` tests for evaluation and API responses.<br>- Write README quick start, benchmark procedure, API examples, and limitations. | - Add preprocessing/WebRTC tests: sample rate, stereo-to-mono, silence input, valid frame length, merge/padding boundaries.<br>- Finalize DSP explanation, figures, and audio-data documentation.<br>- Create demo GIF/video and annotate the VAD visualization. | - Test API against a clean, pause-heavy, noisy, and silence-only file.<br>- Fresh-clone README test.<br>- Each person prepares a 45-second explanation of their work. |

**End-of-day Deliverable:**
- Working CLI and FastAPI endpoint.
- Tests passing.
- README with architecture, methods, results, DSP analysis, limitations, and reproduction commands.
- Demo GIF/video.
- CV-ready contribution bullets.

---

## 🚫 Avoid Scope Creep

These are explicitly **out of scope** for four days:
- ❌ Real-time microphone streaming.
- ❌ Custom ASR training or fine-tuning.
- ❌ Speaker diarization.
- ❌ A complex Streamlit frontend.
- ❌ Docker deployment, unless all core features and documentation are already complete.
- ❌ More than two VAD methods in the main ASR benchmark.
- ❌ A large public dataset or hundreds of manually labeled samples.

*The energy VAD is a DSP learning baseline, not necessarily a third pipeline condition in the final ASR benchmark. The final table should stay focused: no VAD, WebRTC VAD, and Silero VAD.*

---

## 📊 Final Experiment Table

Use this template and fill it only with actual measurements:

| Pipeline | Audio condition | WER (%) ↓ | CER (%) ↓ | RTF ↓ | Speech retained | VAD segments/file | Key observation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full-audio ASR** | Clean | — | — | — | 100% | N/A | Baseline |
| **WebRTC VAD + ASR** | Clean | — | — | — | —% | — | — |
| **Silero VAD + ASR** | Clean | — | — | — | —% | — | — |
| **Full-audio ASR** | Pause-heavy | — | — | — | 100% | N/A | Baseline |
| **WebRTC VAD + ASR** | Pause-heavy | — | — | — | —% | — | — |
| **Silero VAD + ASR** | Pause-heavy | — | — | — | —% | — | — |
| **Full-audio ASR** | Moderate noise | — | — | — | 100% | N/A | Baseline |
| **WebRTC VAD + ASR** | Moderate noise | — | — | — | —% | — | — |
| **Silero VAD + ASR** | Moderate noise | — | — | — | —% | — | — |

### Use aggregate rows too:
| Pipeline | Mean WER ↓ | Mean CER ↓ | Mean RTF ↓ | Mean speech retained ↓ | Boundary behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full-audio ASR** | — | — | — | 100% | N/A |
| **WebRTC VAD + ASR** | — | — | — | —% | — |
| **Silero VAD + ASR** | — | — | — | —% | — |

---

## 🛠️ Minimal Implementation Stack

```text
Python 3.10+
PyTorch + torchaudio       # waveform operations, resampling
soundfile                  # reading/writing WAV
webrtcvad-wheels           # WebRTC VAD
silero-vad                 # neural VAD
faster-whisper             # CTranslate2-powered ASR
jiwer                      # WER/CER
numpy + pandas             # data and results
matplotlib                 # waveform, RMS, spectrogram plots
fastapi + uvicorn          # API
pytest + httpx             # tests
pyyaml                     # experiment configuration
```

*Use `webrtcvad-wheels` rather than the older package where installation compatibility is difficult. If WebRTC installation blocks progress, record it as a setup issue, use a packaged wheel, and make Silero the main VAD path; do not sacrifice the end-to-end deliverable to solve a dependency problem.*

---

## 📂 Suggested Repository Structure

```text
vietspeech-dsp-vad-asr/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── README.md
│   └── test_manifest.csv
├── src/
│   ├── audio.py              # Person B
│   ├── features.py           # shared: RMS, STFT helpers
│   ├── energy_vad.py         # Person A
│   ├── webrtc_vad.py         # Person B
│   ├── silero_vad.py         # Person A
│   ├── segmenter.py          # shared
│   ├── asr.py                # Person A
│   ├── evaluation.py         # Person A
│   ├── pipeline.py           # Person A
│   └── api.py                # Person A
├── scripts/
│   ├── benchmark.py          # Person A
│   └── plot_vad_analysis.py  # Person B
├── tests/
│   ├── test_audio.py         # Person B
│   ├── test_webrtc_vad.py    # Person B
│   ├── test_segmenter.py     # shared
│   ├── test_evaluation.py    # Person A
│   └── test_api.py           # Person A
├── results/
│   ├── benchmark.csv
│   ├── error_analysis.md
│   └── figures/
└── demo/
    └── demo.gif
```

---

## 💼 CV-Ready Contributions

### Person A — Computer Science
- *Built the ASR and evaluation layer of a Vietnamese VAD-gated transcription system using faster-whisper, Silero VAD, FastAPI, and JiWER; benchmarked full-audio, WebRTC-VAD, and neural-VAD pipelines using WER, CER, real-time factor, and segment-level error analysis.*
- *Implemented an RMS-energy VAD baseline and analyzed how framing, energy thresholds, segment padding, and VAD false negatives affected Vietnamese ASR quality and runtime.*

### Person B — Electrical & Computer Engineering
- *Developed the DSP/audio front end for a Vietnamese ASR pipeline, including 16 kHz mono PCM conversion, 20 ms frame processing, RMS-energy analysis, WebRTC VAD integration, and speech-segment postprocessing.*
- *Compared frame-based VAD aggressiveness settings using waveform/spectrogram diagnostics and annotated boundary cases, identifying trade-offs among noise rejection, speech clipping, and ASR compute reduction.*

---

## 🎤 Interview Explanation

### A concise explanation both team members can use:
> *"We built an offline Vietnamese ASR benchmark that compares transcription of the full audio with transcription after WebRTC VAD or Silero VAD segmentation. We standardized recordings to 16 kHz mono, used frame-level analysis to inspect energy and speech boundaries, and sent detected speech regions to faster-whisper. We measured WER, CER, real-time factor, speech retained, and segmentation failures across clean, pause-heavy, and moderately noisy Vietnamese recordings. The comparison let us study the practical trade-off: aggressive VAD can reduce unnecessary ASR processing, but it can also clip quiet speech or word boundaries."*

---

## 📚 References
1. https://github.com/wiseman/py-webrtcvad/blob/master/README.rst
2. https://github.com/snakers4/silero-vad
3. https://github.com/SYSTRAN/faster-whisper
4. https://github.com/snakers4/silero-vad/issues/2
5. https://runthisai.com/en/blog/faster-whisper-guide
6. https://raw.githubusercontent.com/NeverSight/skills_feed/refs/heads/main/data/skills-md/theplasmak/faster-whisper/faster-whisper/SKILL.md
7. https://runthisai.com/en/tool/faster-whisper
8. https://doc.rapida.ai/opensource/services/assistant-api/vad/silero
9. https://openapps.pro/packages/faster-whisper
10. https://github.com/AIXerum/faster-whisper
11. https://github.com/bungerr/faster-whisper-3
12. https://github.com/deeenancy/faster-whisper-CT2
13. https://github.com/openai/whisper/discussions/937
14. https://github.com/BBC-Esq/Faster-Whisper-Transcriber
15. https://github.com/snakers4/silero-vad/releases
