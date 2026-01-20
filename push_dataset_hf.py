from huggingface_hub import (
    HfApi,
    create_repo,
    upload_folder,
)
from huggingface_hub.utils import RepositoryNotFoundError

# ==============================
# CONFIG
# ==============================
REPO_ID = "redha24/calvin_task_D_D_pick_push"
REPO_TYPE = "dataset"
LOCAL_FOLDER = "/mnt/d/LeRobot/calvin_task_D_D_pick_push"
PRIVATE = False
"""
REPO_ID = "redha24/calvin_small_debug_train"
REPO_TYPE = "dataset"
LOCAL_FOLDER = "calvin_small_debug_train"
PRIVATE = False
"""
# ==============================
# PUSH SCRIPT
# ==============================
api = HfApi()

# --------------------------------------------------
# 1. Delete repo if it exists
# --------------------------------------------------
try:
    print(f"🗑️ Deleting existing repo: {REPO_ID}")
    api.delete_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print("✅ Repo deleted")
except RepositoryNotFoundError:
    print("ℹ️ Repo does not exist yet (nothing to delete)")

# --------------------------------------------------
# 2. Recreate repo
# --------------------------------------------------
print(f"📦 Creating repo: {REPO_ID}")
create_repo(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    private=PRIVATE,
    exist_ok=False,
)
print("✅ Repo created")

# --------------------------------------------------
# 3. Upload local dataset folder
# --------------------------------------------------
print(f"⬆️ Uploading folder: {LOCAL_FOLDER}")
upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)
print("✅ Upload complete")

print("\n🎉 Dataset pushed cleanly to Hugging Face!")
print(f"🔗 https://huggingface.co/datasets/{REPO_ID}")
