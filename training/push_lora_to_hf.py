from huggingface_hub import create_repo, upload_folder
from pathlib import Path

HF_USERNAME = "An-di"
REPO_NAME = "qwen2_5_3b_jokes_lora"

REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
LOCAL_ADAPTER_DIR = Path("../models/qwen_lora_jokes")

if not LOCAL_ADAPTER_DIR.exists():
    raise SystemExit(f"Local adapter dir not found: {LOCAL_ADAPTER_DIR}")

print(f"Creating private repo: {REPO_ID}")
create_repo(
    REPO_ID,
    private=True,
    exist_ok=True,
)

print(f"Uploading folder {LOCAL_ADAPTER_DIR} ...")
upload_folder(
    repo_id=REPO_ID,
    folder_path=str(LOCAL_ADAPTER_DIR),
)

print(f"Done. Adapter is now on the Hub at: {REPO_ID}")
