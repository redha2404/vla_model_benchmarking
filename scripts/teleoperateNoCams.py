from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


def main():
    robot_config = SO101FollowerConfig(
        port="/dev/tty.usbmodem58FA1025861",
        id="FOLLOWER",
    )
    robot = SO101Follower(robot_config)

    teleop_config = SO100LeaderConfig(
        port="/dev/tty.usbmodem58FA1026511",
        id="LEADER",
    )
    teleop = SO100Leader(teleop_config)

    print("Connecting robot and teleoperator...")
    robot.connect()
    teleop.connect()

    print("Teleoperation started")
    try:
        while True:
            action = teleop.get_action()
            robot.send_action(action)

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")

    finally:
        teleop.disconnect()
        robot.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()