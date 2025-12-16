from pathlib import Path
import logging

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.utils.utils import init_logging


#our config
EPISODE_INDICES = [30, 31, 32]

DATASET_REPO_ID = "multiobject-noClutter"
DATASET_ROOT = Path("/Users/darius/Intel/lerobot/multiobject-noClutter")

PUSH_TO_HUB = False

def main():
    init_logging()
    logging.info("Starting delete-episodes operation")

    # Load existing dataset
    dataset = LeRobotDataset(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
    )

    logging.info(
        f"Loaded dataset with {dataset.meta.total_episodes} episodes"
    )

    # Output directory
    output_dir = DATASET_ROOT

    #Careful here, we had this issue at my place
    # When new_repo_id is None, delete_episodes expects the *original dataset to live under <root>_old
    dataset.root = Path(str(dataset.root) + "_old")

    #delete
    new_dataset = delete_episodes(
        dataset=dataset,
        episode_indices=EPISODE_INDICES,
        output_dir=output_dir,
        repo_id=DATASET_REPO_ID,
    )

    logging.info(
        f"Delete complete. New episode count: {new_dataset.meta.total_episodes}"
    )

    if PUSH_TO_HUB:
        logging.info("Pushing cleaned dataset to Hugging Face Hub")
        new_dataset.push_to_hub()

    logging.info("Done.")


if __name__ == "__main__":
    main()