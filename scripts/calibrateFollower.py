from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

def main():
    follower_config = SO101FollowerConfig(
        port="/dev/tty.usbmodem58FA1025861",
        id="FOLLOWER",
    )

    follower = SO101Follower(follower_config)

    print("Connecting to SO101 follower...")
    follower.connect(calibrate=False)

    print("Running calibration for follower...")
    follower.calibrate()

    print("Disconnecting follower.")
    follower.disconnect()


if __name__ == "__main__":
    main()