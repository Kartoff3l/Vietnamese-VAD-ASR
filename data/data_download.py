from huggingface_hub import snapshot_download

repo_id = "VietAudio-team/Vietnamese-asr-leaderboard"

snapshot_download(
    repo_id=repo_id,
    allow_patterns="bud500/*",
    repo_type="dataset",
    local_dir="./data/eval"
)