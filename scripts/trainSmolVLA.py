import time
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.utils import init_logging
from lerobot.utils.random_utils import set_seed
from lerobot.utils.import_utils import register_third_party_plugins


DATASET_REPO_ID = "dariusss04/multicolor-cube"
DATASET_ROOT = Path("/Users/darius/Intel/lerobot/multicolor-cube")

OUTPUT_DIR = Path("/Users/darius/Intel/lerobot/VLAs/smolVLA/multicolor-cube-smolvla")

BATCH_SIZE = 1            
STEPS = 500               
LR = 1e-4
WEIGHT_DECAY = 1e-6
LOG_EVERY = 10
SAVE_EVERY = 200
SEED = 42
NUM_WORKERS = 0
GRAD_CLIP_NORM = 1.0
HORIZON = 50   


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

    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
    )
    
    cfg.pad_language_to = "max_length"          
    cfg.tokenizer_max_length = 48

    cfg.n_action_steps = HORIZON
    cfg.chunk_size = HORIZON

    cfg.device = str(device) 
    cfg.use_amp = False    
    cfg.freeze_vision_encoder = True
    cfg.train_expert_only = True
    cfg.train_state_proj = True
    cfg.load_vlm_weights = False  

    fps = ds0.meta.fps
    delta_timestamps = {
        "action": [i / fps for i in range(HORIZON)],
    }
    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )

    preprocessor, postprocessor = make_smolvla_pre_post_processors(cfg, dataset_stats=ds0.meta.stats)

    policy = SmolVLAPolicy(cfg).to(device).train()

    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=cfg.optimizer_betas,
        eps=cfg.optimizer_eps,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_amp = (device.type == "cuda") and cfg.use_amp
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

        if "action_is_pad" in batch and "actions_id_pad" not in batch:
            batch["actions_id_pad"] = batch["action_is_pad"]

        batch = move_tensors_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            loss, loss_dict = policy.forward(batch)  

        loss.backward()

        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        step += 1

        if step % LOG_EVERY == 0:
            elapsed = time.perf_counter() - start_time
            logging.info(
                f"step={step}/{STEPS} loss={loss.item():.4f} "
                f"lang_len={batch['observation.language.tokens'].shape[-1]} "
                f"time_s={elapsed:.1f}"
            )

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