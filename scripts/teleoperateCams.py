from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def main():
    
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=0,
            width=640,
            height=480,
            fps=30,
            warmup_s=2,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=1,
            width=640,
            height=480,
            fps=30,
            warmup_s=2,
        ),
    }

    robot_config = SO101FollowerConfig(
        port="/dev/tty.usbmodem58FA1025861",
        id="FOLLOWER",
        cameras=cameras,
    )
    robot = SO101Follower(robot_config)
    
    teleop_config = SO101LeaderConfig(
        port="/dev/tty.usbmodem58FA1026511",
        id="LEADER",
    )
    teleop = SO101Leader(teleop_config)

    #display_data=true
    init_rerun(session_name="teleoperation")

    print("Connecting robot and teleoperator...")
    robot.connect()
    teleop.connect()

    print("Teleoperation with cameras started")
    try:
        while True:
            action = teleop.get_action()

            robot.send_action(action)

            obs = robot.get_observation()
           
            # Live visualization
            log_rerun_data(
                observation=obs,
                action=action,
            )

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")

    finally:
        teleop.disconnect()
        robot.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()