from huggingface_hub import HfApi, upload_folder
from pathlib import Path

REPO_ID = "An-di/qwen2_5_3b_jokes_lora"
LOCAL_DIR = Path("../models/qwen_lora_jokes")

api = HfApi()

# Upload new files (overwrites same paths, leaves extra remote files intact)
upload_folder(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=str(LOCAL_DIR),
)

# Delete remote files that are not present locally
remote_files = set(api.list_repo_files(repo_id=REPO_ID, repo_type="model"))
local_files = set(
    str(p.relative_to(LOCAL_DIR)).replace("\\", "/")
    for p in LOCAL_DIR.rglob("*")
    if p.is_file()
)

to_delete = sorted(remote_files - local_files)

for path_in_repo in to_delete:
    api.delete_file(repo_id=REPO_ID, repo_type="model", path_in_repo=path_in_repo)

print(f"Done. Deleted {len(to_delete)} leftover files.")
