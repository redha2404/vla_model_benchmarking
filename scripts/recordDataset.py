from pathlib import Path

from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts

from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot.scripts.lerobot_record import record_loop

#Our config
DATASET_REPO_ID = "dariusss04/multiobject-noClutter"
DATASET_ROOT = Path("/Users/darius/Intel/lerobot/multiobject-noClutter")

TASK_DESCRIPTION = "Pick up the cube and place it in the grey container"
NUM_EPISODES = 1

RESUME = True
PUSH_TO_HUB = True

FPS = 30

FOLLOWER_PORT = "/dev/tty.usbmodem58FA1025861"
LEADER_PORT = "/dev/tty.usbmodem58FA1026511"

def main():
    register_third_party_plugins()
    init_logging()

    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=0,
            width=640,
            height=480,
            fps=30,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=1,
            width=640,
            height=480,
            fps=30,
        ),
    }

    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id="FOLLOWER",
        cameras=cameras,
    )
    robot = make_robot_from_config(robot_cfg)

    teleop_cfg = SO101LeaderConfig(
        port=LEADER_PORT,
        id="LEADER",
    )
    teleop = make_teleoperator_from_config(teleop_cfg)

    teleop_action_processor, robot_action_processor, robot_obs_processor = (
        make_default_processors()
    )

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_obs_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=True,
        ),
    )

    #dataset load/resume
    if RESUME:
        dataset = LeRobotDataset(
            DATASET_REPO_ID,
            root=DATASET_ROOT,
        )

        dataset.start_image_writer(
            num_processes=0,
            num_threads=4 * len(cameras),
        )

        sanity_check_dataset_robot_compatibility(
            dataset,
            robot,
            FPS,
            dataset_features,
        )
    else:
        dataset = LeRobotDataset.create(
            DATASET_REPO_ID,
            FPS,
            root=DATASET_ROOT,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
        )

    robot.connect()
    teleop.connect()

    listener, events = init_keyboard_listener()

    #recording loop
    with VideoEncodingManager(dataset):
        recorded = 0
        while recorded < NUM_EPISODES and not events["stop_recording"]:
            log_say(
                f"Recording episode {dataset.meta.total_episodes}",
                blocking=False,
            )

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_obs_processor,
                teleop=teleop,
                dataset=dataset,
                control_time_s=60,
                single_task=TASK_DESCRIPTION,
                display_data=False,
            )

            dataset.save_episode()
            recorded += 1

    robot.disconnect()
    teleop.disconnect()

    if listener is not None:
        listener.stop()

    #push to huggingface
    if PUSH_TO_HUB:
        dataset.push_to_hub()

    log_say("Recording finished", blocking=True)


if __name__ == "__main__":
    main()