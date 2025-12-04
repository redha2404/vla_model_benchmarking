from pathlib import Path
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

# Helpers
def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]

# Settings
DATASET_ID = "lerobot/aloha_sim_insertion_human"

OUTPUT_DIR = Path("/Users/darius/Intel/projects/vlaBench/models/diffusion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load dataset metadata
metadata = LeRobotDatasetMetadata(DATASET_ID)

# Convert raw features to policy input/output formats
features = dataset_to_policy_features(metadata.features)

output_features = {
    k: f for k, f in features.items() if f.type is FeatureType.ACTION
}
input_features = {
    k: f for k, f in features.items() if k not in output_features
}

# Diffusion Config + Policy
cfg = DiffusionConfig(
    input_features=input_features,
    output_features=output_features,
)

policy = DiffusionPolicy(cfg)
policy.to(device)
policy.train()

preprocessor, postprocessor = make_pre_post_processors(
    cfg, dataset_stats=metadata.stats
)

delta_timestamps = {
    "observation.state": make_delta_timestamps(
        cfg.observation_delta_indices, metadata.fps
    ),
    "action": make_delta_timestamps(cfg.action_delta_indices, metadata.fps),
}

# Add timestamps for image features 
for key in cfg.image_features.keys():
    delta_timestamps[key] = make_delta_timestamps(
        cfg.observation_delta_indices, metadata.fps
    )


dataset = LeRobotDataset(DATASET_ID, delta_timestamps=delta_timestamps)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    drop_last=True,
    pin_memory=(device.type != "cpu")
)

optimizer = cfg.get_optimizer_preset().build(policy.parameters())

#Train loop
training_steps = 200  
log_every = 10

step = 0
done = False

print("Starting training...")

while not done:
    for batch in dataloader:
        batch = preprocessor(batch)

        loss, _ = policy.forward(batch)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if step % log_every == 0:
            print(f"[step {step}] loss = {loss.item():.4f}")

        step += 1
        if step >= training_steps:
            done = True
            break


#Save model
policy.save_pretrained(OUTPUT_DIR)
preprocessor.save_pretrained(OUTPUT_DIR)
postprocessor.save_pretrained(OUTPUT_DIR)

print(f"Training complete. Model saved to {OUTPUT_DIR}")