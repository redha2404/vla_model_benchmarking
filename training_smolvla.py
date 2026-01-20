# On Kaggle (parameters : GPU P100, keep files, access to Internet)

# Install requirements
!git clone https://github.com/huggingface/lerobot.git
!cd lerobot && pip install -e ".[smolvla]"
!pip install num2words

# Login
import os
os.environ["WANDB_API_KEY"] = "KEY"
import wandb
wandb.login()
from huggingface_hub import login
login(token="TOKEN")

# Preparation
from huggingface_hub import HfApi
api = HfApi()
info = api.dataset_info("redha24/calvin_task_D_D_pick_push")
print("Dataset found:", info.id)

from huggingface_hub import HfApi
hub_api = HfApi()
hub_api.create_tag("redha24/calvin_task_D_D_pick_push", tag="v3.0", repo_type="dataset")

# Optional : delete folder to retry
import shutil
from pathlib import Path

OUTPUT_DIR = Path("outputs/train/smolvla_calvin_pick_push")

print("Current working directory:", Path.cwd())
print("Output dir resolved to:", OUTPUT_DIR.resolve())

if OUTPUT_DIR.exists():
    print(f"🗑️ Removing existing output dir: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)
else:
    print("✅ No previous output dir found")
  
# Training
!lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=redha24/calvin_task_D_D_pick_push \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --batch_size=16 \
  --steps=8000 \
  --output_dir=outputs/train/smolvla_calvin_pick_push \
  --save_checkpoint=true \
  --save_freq=500 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=redha24/smolvla_calvin_pick_push

# Create a model repo and push checkpoints (here 007500)
from huggingface_hub import HfApi
api = HfApi()

api.create_repo(
    repo_id="redha24/smolvla_calvin_pick_push",
    repo_type="model",
    exist_ok=True,
)
api.upload_folder(
    folder_path="outputs/train/smolvla_calvin_pick_push/checkpoints/007500/pretrained_model",
    repo_id="redha24/smolvla_calvin_pick_push",
    repo_type="model",
)

