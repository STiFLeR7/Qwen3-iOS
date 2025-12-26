from huggingface_hub import hf_hub_download
from pathlib import Path

REPO_ID = "STiFLeR7/qwen3-ios-executorch"
TARGET_DIR = Path("ios/Models/qwen3_0_6b")

TARGET_DIR.mkdir(parents=True, exist_ok=True)

files = [
    "qwen3_0.6B_model.pte",
    "0.6B_config.json",
]

for f in files:
    hf_hub_download(
        repo_id=REPO_ID,
        filename=f,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
    )

print("Model artifacts downloaded successfully.")
