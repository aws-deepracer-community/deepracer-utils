import sys
import deepracer.boto3_enhancer


def main():
    if "install-cli" in sys.argv:
        force = True if "--force" in sys.argv else False

        deepracer.boto3_enhancer.install_deepracer_cli(force)
    elif "remove-cli" in sys.argv:
        deepracer.boto3_enhancer.remove_deepracer_cli()


if __name__ == "__main__":
    main()
