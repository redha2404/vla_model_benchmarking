import time
import math
import logging
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader

from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import init_logging
from lerobot.utils.random_utils import set_seed
from lerobot.utils.import_utils import register_third_party_plugins


DATASET_LOCAL_ROOT = Path("/kaggle/working/multicolor-cube-hf")
OUTPUT_DIR = Path("/kaggle/working/VLAs/ACT/multicolor-cube-act")

HF_REPO_ID = "dariusss04/act-multicolour-cube"

BATCH_SIZE = 8
STEPS = 60_000
LR = 5e-5
WEIGHT_DECAY = 1e-4
LOG_EVERY = 50
SAVE_EVERY = 2000
SEED = 42
NUM_WORKERS = 0
GRAD_CLIP_NORM = 1.0



def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=(device.type == "cuda"))
    return batch


def squeeze_singleton_time_dim(batch):
    for k in [
        "observation.state",
        "observation.state_is_pad",
        "observation.images.front_is_pad",
        "observation.images.wrist_is_pad",
    ]:
        if k in batch and torch.is_tensor(batch[k]) and batch[k].ndim > 1 and batch[k].shape[1] == 1:
            batch[k] = batch[k].squeeze(1)
    return batch


class SafeLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.n = len(ds)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = int(idx) % self.n
        for _ in range(50):
            try:
                return self.ds[i]
            except Exception:
                i = (i + 1) % self.n
        raise RuntimeError("Too many failed samples")


def save_all(policy, pre, post, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    pre.save_pretrained(out_dir / "preprocessor")
    post.save_pretrained(out_dir / "postprocessor")


def push_to_hf(local_dir: Path, repo_subdir: str):
    api = HfApi()
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=HF_REPO_ID,
        repo_type="model",
        path_in_repo=repo_subdir,
        commit_message=f"Upload {repo_subdir}",
    )


def main():
    register_third_party_plugins()
    init_logging()
    set_seed(SEED)

    device = get_device()
    logging.info(f"Using device: {device}")

    ds0 = LeRobotDataset(str(DATASET_LOCAL_ROOT))
    logging.info(f"Episodes: {ds0.num_episodes}, Frames: {ds0.num_frames}")

    features = dataset_to_policy_features(ds0.meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        use_vae=True,
    )

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, ds0.meta.fps),
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, ds0.meta.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, ds0.meta.fps)
        for k in cfg.image_features
    }

    base_ds = LeRobotDataset(str(DATASET_LOCAL_ROOT), delta_timestamps=delta_timestamps)
    dataset = SafeLeRobotDataset(base_ds)

    pre, post = make_pre_post_processors(cfg, dataset_stats=ds0.meta.stats)

    policy = ACTPolicy(cfg).to(device).train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Starting ACT training...")
    start_time = time.perf_counter()
    step = 0
    dl_iter = iter(dataloader)

    while step < STEPS:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        batch = pre(batch)
        batch = squeeze_singleton_time_dim(batch)
        batch = move_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy(batch)

        if not torch.isfinite(loss):
            logging.error(f"NaN/Inf loss at step {step}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        step += 1

        if step % LOG_EVERY == 0:
            logging.info(
                f"step={step}/{STEPS} loss={loss.item():.4f} "
                f"time_s={time.perf_counter() - start_time:.1f}"
            )

        if step % SAVE_EVERY == 0:
            ckpt_dir = OUTPUT_DIR / f"checkpoint_{step:06d}"
            save_all(policy, pre, post, ckpt_dir)
            print(f"Saved checkpoint at step {step}")
            push_to_hf(ckpt_dir, ckpt_dir.name)
            print(f"Pushed checkpoint {ckpt_dir.name} to Hugging Face")

    final_dir = OUTPUT_DIR / "final"
    save_all(policy, pre, post, final_dir)
    print("Saved FINAL model")
    push_to_hf(final_dir, "final")
    print("Pushed FINAL model to Hugging Face")

    logging.info("Training finished.")


if __name__ == "__main__":
    main()