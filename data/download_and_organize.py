import os
import io
import sys
import shutil
import argparse
import soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download, HfFileSystem
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

REPO_ID = "VietAudio-team/Vietnamese-asr-leaderboard"
REPO_TYPE = "dataset"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "eval"))

# 4 condition cân bằng (mỗi nhóm đúng 6 mẫu chuẩn)
CATEGORIES = {
    "moderate_noise": [
        "bud500_0", "bud500_1", "bud500_2", "bud500_3", "bud500_4", "bud500_5"
    ],
    "clean": [
        "bud500_13", "VIVOSDEV07_098", "VIVOSDEV07_168", "VIVOSDEV01_R099", "VIVOSDEV02_R079", "VIVOSDEV03_R113"
    ],
    "pause_heavy": [
        "fosd_1", "vimd_1", "common_voice_6", "common_voice_17", "common_voice_26", "common_voice_49"
    ],
    "short_utterance": [
        "common_voice_0", "common_voice_1", "common_voice_4", "common_voice_78", "common_voice_94", "common_voice_96"
    ]
}

def get_subset_for_target(target: str) -> str:
    if target.startswith("bud500_"):
        return "bud500"
    elif target.startswith("common_voice_"):
        return "common_voice"
    elif target.startswith("fosd_"):
        return "fosd"
    elif target.startswith("vimd_"):
        return "vimd"
    elif target.startswith("vivos_") or target.startswith("VIVOS"):
        return "vivos"
    return ""

def stream_extract_vimd(target_id="vimd_1", out_category="pause_heavy"):
    """
    Trích xuất vimd_1 trực tiếp từ Row Group 0 qua network stream (chỉ ~84MB)
    thay vì phải tải toàn bộ file 12.5GB về máy.
    """
    out_dir = os.path.join(DATA_DIR, out_category)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{target_id}.wav")

    if os.path.exists(out_path):
        print(f"[*] {target_id}.wav đã có sẵn tại {out_path}")
        return True

    print(f"\n[+] Đang stream trích xuất '{target_id}' từ vimd (chỉ đọc Row Group 0 ~84MB qua mạng)...")
    try:
        fs = HfFileSystem()
        remote_path = f"datasets/{REPO_ID}/vimd/validation-00000-of-00001.parquet"
        with fs.open(remote_path, "rb") as f:
            pf = pq.ParquetFile(f)
            rg0 = pf.read_row_group(0)
            ids = rg0["id"].to_pylist()
            audio_col = rg0["audio"]
            for i, id_val in enumerate(ids):
                if id_val == target_id:
                    raw = audio_col[i].as_py()
                    data, sr = sf.read(io.BytesIO(raw["bytes"]))
                    sf.write(out_path, data, sr)
                    print(f"  -> [{out_category}] Đã lưu: {target_id}.wav (sr={sr}Hz, len={len(data)/sr:.2f}s)")
                    return True
    except Exception as e:
        print(f"[!] Lỗi khi stream vimd: {e}")
        return False
    return False

def download_subsets(subsets_to_download: set):
    """Tải các file parquet cần thiết (bỏ qua vimd vì dùng streaming)."""
    for subset in sorted(subsets_to_download):
        if subset == "vimd":
            continue
        parquet_path = os.path.join(DATA_DIR, subset, "validation-00000-of-00001.parquet")
        if os.path.exists(parquet_path):
            print(f"[*] {subset} đã có sẵn tại: {parquet_path}")
            continue

        print(f"\n[+] Đang tải subset '{subset}' từ {REPO_ID}...")
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=f"{subset}/*",
            repo_type=REPO_TYPE,
            local_dir=DATA_DIR
        )
        print(f"[+] Tải thành công '{subset}'.")

def cleanup_cache_and_parquets():
    """Xóa các thư mục parquet và cache sau khi trích xuất âm thanh xong."""
    print("\n" + "=" * 60)
    print(" DỌN DẸP FILE PARQUET VÀ CACHE ĐỂ GIẢI PHÓNG DUNG LƯỢNG")
    print("=" * 60)
    to_delete = ["bud500", "common_voice", "fosd", "vivos", "vimd", ".cache"]
    freed = 0
    for item in to_delete:
        p = os.path.join(DATA_DIR, item)
        if os.path.exists(p):
            for root, dirs, files in os.walk(p):
                for f in files:
                    try:
                        freed += os.path.getsize(os.path.join(root, f))
                    except Exception:
                        pass
            shutil.rmtree(p)
            print(f"  -> Đã xóa: {item}")
    print(f"[✓] Đã giải phóng: {freed / (1024**2):.2f} MB dung lượng đĩa.")

def run_pipeline(clean_after: bool = True):
    # Tạo thư mục categories
    for cat in CATEGORIES:
        os.makedirs(os.path.join(DATA_DIR, cat), exist_ok=True)

    # Gom target theo subset
    subset_targets = {}
    target_to_cat = {}
    for cat, targets in CATEGORIES.items():
        for t in targets:
            sub = get_subset_for_target(t)
            if not sub:
                continue
            subset_targets.setdefault(sub, []).append(t)
            target_to_cat[t] = cat

    # Tải các parquet nhỏ
    download_subsets(set(subset_targets.keys()))

    print("\n" + "=" * 60)
    print(" BẮT ĐẦU TRÍCH XUẤT VÀ PHÂN LOẠI AUDIO VÀO THƯ MỤC")
    print("=" * 60)

    # Xử lý vimd riêng bằng streaming
    if "vimd" in subset_targets:
        for t in subset_targets["vimd"]:
            stream_extract_vimd(target_id=t, out_category=target_to_cat[t])

    # Xử lý các subset từ parquet
    for subset, targets in subset_targets.items():
        if subset == "vimd":
            continue
        parquet_path = os.path.join(DATA_DIR, subset, "validation-00000-of-00001.parquet")
        if not os.path.exists(parquet_path):
            continue

        print(f"\n[*] Đang quét file: {subset} (mục tiêu: {targets})")
        pf = pq.ParquetFile(parquet_path)
        found_targets = set()

        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            ids = rg["id"].to_pylist()
            audio_col = rg["audio"]

            for i, current_id in enumerate(ids):
                matched = None
                if current_id in targets:
                    matched = current_id
                else:
                    path_val = str(audio_col[i]["path"].as_py() or "")
                    for t in targets:
                        if t in path_val:
                            matched = t
                            break

                if matched and matched not in found_targets:
                    found_targets.add(matched)
                    cat = target_to_cat[matched]
                    raw = audio_col[i].as_py()
                    data, sr = sf.read(io.BytesIO(raw["bytes"]))
                    out_name = f"{matched}.wav"
                    out_p = os.path.join(DATA_DIR, cat, out_name)
                    sf.write(out_p, data, sr)
                    print(f"  -> [{cat}] Đã lưu: {out_name} (sr={sr}Hz, len={len(data)/sr:.2f}s)")

            if len(found_targets) == len(targets):
                break

    # Dọn dẹp cache và parquet nếu bật
    if clean_after:
        cleanup_cache_and_parquets()

    # Cập nhật manifest tổng hợp
    update_manifest()

def update_manifest():
    records = []
    for cat in CATEGORIES:
        cat_p = os.path.join(DATA_DIR, cat)
        if not os.path.exists(cat_p):
            continue
        for f in sorted(os.listdir(cat_p)):
            if f.endswith(".wav"):
                fp = os.path.join(cat_p, f)
                info = sf.info(fp)
                rel_p = os.path.join("eval", cat, f).replace("\\", "/")
                records.append({
                    "id": os.path.splitext(f)[0],
                    "condition": cat,
                    "audio_path": rel_p,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "duration_seconds": round(info.duration, 3)
                })

    df = pd.DataFrame(records)
    manifest_p = os.path.join(DATA_DIR, "filtered_manifest.csv")
    df.to_csv(manifest_p, index=False, encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"[✓] Đã tạo thành công và cập nhật manifest với {len(df)} file audio!")
    print(f"[✓] File manifest lưu tại: {manifest_p}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải và sắp xếp các file audio theo condition từ Vietnamese ASR Leaderboard")
    parser.add_argument("--no-clean", action="store_true", help="Không xóa các file parquet và cache sau khi trích xuất")
    args = parser.parse_args()

    run_pipeline(clean_after=not args.no_clean)
