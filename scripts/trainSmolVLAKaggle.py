import time
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.utils import init_logging
from lerobot.utils.random_utils import set_seed
from lerobot.utils.import_utils import register_third_party_plugins


DATASET_LOCAL_ROOT = Path("/kaggle/working/multicolor-cube-hf")
OUTPUT_DIR = Path("/kaggle/working/VLAs/smolVLA/multicolor-cube-smolvla")
HF_REPO_ID = "dariusss04/smolVLA-multicolor-cube"

BATCH_SIZE = 16
STEPS = 60_000
LR = 1e-4
WEIGHT_DECAY = 1e-10
LOG_EVERY = 50
SAVE_EVERY = 2000
SEED = 42
NUM_WORKERS = 0
GRAD_CLIP_NORM = 10.0
PIN_MEMORY = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class SafeLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, ds, max_tries=50):
        self.ds = ds
        self.n = len(ds)
        self.max_tries = max_tries

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = int(idx) % self.n
        for _ in range(self.max_tries):
            try:
                return self.ds[i]
            except Exception:
                i = (i + 1) % self.n
        raise RuntimeError("Too many failed samples in dataset __getitem__")


def build_action_chunk_delta_timestamps(fps: int, horizon: int):
    return {"action": [i / fps for i in range(horizon)]}


def main():
    register_third_party_plugins()
    init_logging()
    set_seed(SEED)

    device = get_device()
    logging.info(f"Using device: {device}")

    ds0 = LeRobotDataset(str(DATASET_LOCAL_ROOT))
    assert str(ds0.root) == str(DATASET_LOCAL_ROOT), (ds0.root, DATASET_LOCAL_ROOT)
    assert ds0.num_frames == 12371, ds0.num_frames
    assert ds0.meta.stats["observation.images.front"]["count"] == [12371]
    assert ds0.meta.stats["observation.images.wrist"]["count"] == [12371]
    logging.info("✅ Training will use LOCAL snapshot + correct stats")

    features = dataset_to_policy_features(ds0.meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        device=str(device),
    )
    cfg.load_vlm_weights = True
    cfg.pad_language_to = "max_length"
    cfg.tokenizer_max_length = 48

    H = int(cfg.n_action_steps)
    delta_timestamps = build_action_chunk_delta_timestamps(ds0.meta.fps, H)

    base_ds = LeRobotDataset(str(DATASET_LOCAL_ROOT), delta_timestamps=delta_timestamps)
    dataset = SafeLeRobotDataset(base_ds)

    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=ds0.meta.stats)

    policy = SmolVLAPolicy(cfg).to(device).train()

    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=LR,
        betas=cfg.optimizer_betas,
        eps=cfg.optimizer_eps,
        weight_decay=WEIGHT_DECAY,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=(PIN_MEMORY and device.type == "cuda"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    logging.info("Starting SmolVLA training...")
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

        if "action_is_pad" in batch and "actions_id_pad" not in batch:
            batch["actions_id_pad"] = batch["action_is_pad"]

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            loss, _ = policy.forward(batch)

        if not torch.isfinite(loss):
            logging.error(f"NaN/Inf loss at step {step}. Stopping.")
            break

        loss.backward()

        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        step += 1

        if step % LOG_EVERY == 0:
            lang_len = int(batch["observation.language.tokens"].shape[-1]) if "observation.language.tokens" in batch else None
            logging.info(
                f"step={step}/{STEPS} loss={loss.item():.4f}"
                + (f" lang_len={lang_len}" if lang_len is not None else "")
                + f" time_s={time.perf_counter() - start_time:.1f}"
            )

        if step % SAVE_EVERY == 0:
            ckpt_dir = OUTPUT_DIR / f"checkpoint_{step:06d}"
            save_all(policy, pre, post, ckpt_dir)
            logging.info(f"Saved checkpoint at step {step} -> {ckpt_dir}")
            push_to_hf(ckpt_dir, ckpt_dir.name)
            logging.info(f"Pushed checkpoint {ckpt_dir.name} to Hugging Face")

    final_dir = OUTPUT_DIR / "final"
    save_all(policy, pre, post, final_dir)
    logging.info("Saved FINAL model")
    push_to_hf(final_dir, "final")
    logging.info("Pushed FINAL model to Hugging Face")
    logging.info("Training finished.")


if __name__ == "__main__":
    main()