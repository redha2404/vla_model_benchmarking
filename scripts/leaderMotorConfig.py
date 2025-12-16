from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

def main():
    leader_config = SO101LeaderConfig(
        port="/dev/tty.usbmodem58FA1026511",
        id="LEADER",
    )

    leader = SO101Leader(leader_config)

    print("Connecting to SO101 leader...")
    leader.connect()

    print("Running motor setup for leader...")
    leader.setup_motors()

    print("Disconnecting leader.")
    leader.disconnect()


if __name__ == "__main__":
    main()