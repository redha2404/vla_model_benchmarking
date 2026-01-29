import time
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import init_logging
from lerobot.utils.random_utils import set_seed
from lerobot.utils.import_utils import register_third_party_plugins


DATASET_REPO_ID = "dariusss04/multicolor-cube"
DATASET_ROOT = Path("/Users/darius/Intel/lerobot/multicolor-cube")

OUTPUT_DIR = Path("/Users/darius/Intel/lerobot/VLAs/ACT/multicolor-cube-act")

BATCH_SIZE = 2
STEPS = 20_000
LR = 1e-4
WEIGHT_DECAY = 1e-4
LOG_EVERY = 50
SAVE_EVERY = 2000
SEED = 42
NUM_WORKERS = 0
GRAD_CLIP_NORM = 1.0


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_tensors_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=(device.type == "cuda"))
    return batch


def squeeze_singleton_time_dim(batch):
    if "observation.state" in batch and batch["observation.state"].ndim == 3 and batch["observation.state"].shape[1] == 1:
        batch["observation.state"] = batch["observation.state"].squeeze(1)
    if "observation.state_is_pad" in batch and batch["observation.state_is_pad"].ndim == 2 and batch["observation.state_is_pad"].shape[1] == 1:
        batch["observation.state_is_pad"] = batch["observation.state_is_pad"].squeeze(1)
    return batch


def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def main():
    register_third_party_plugins()
    init_logging()
    set_seed(SEED)

    device = get_device()
    logging.info(f"Using device: {device}")

    ds0 = LeRobotDataset(DATASET_REPO_ID, root=DATASET_ROOT)

    features = dataset_to_policy_features(ds0.meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        use_vae=False,
    )

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, ds0.meta.fps),
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, ds0.meta.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, ds0.meta.fps)
        for k in cfg.image_features
    }

    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=ds0.meta.stats)

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

    use_amp = device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    start_time = time.perf_counter()
    step = 0
    dl_iter = iter(dataloader)

    while step < STEPS:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        batch = preprocessor(batch)
        batch = squeeze_singleton_time_dim(batch)
        batch = move_tensors_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            loss, _ = policy.forward(batch)

        loss.backward()

        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        step += 1

        if step % LOG_EVERY == 0:
            elapsed = time.perf_counter() - start_time
            logging.info(f"step={step}/{STEPS} loss={loss.item():.4f} time_s={elapsed:.1f}")

        if step % SAVE_EVERY == 0:
            ckpt_dir = OUTPUT_DIR / f"checkpoint_{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ckpt_dir)
            preprocessor.save_pretrained(ckpt_dir / "preprocessor")
            postprocessor.save_pretrained(ckpt_dir / "postprocessor")

    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir / "preprocessor")
    postprocessor.save_pretrained(final_dir / "postprocessor")

    logging.info("Training finished.")


if __name__ == "__main__":
    main()