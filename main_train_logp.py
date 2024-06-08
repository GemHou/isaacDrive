import torch

from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu


def main():
    isaac_drive_env = IsaacDriveEnv(device=DEVICE)

    print("Finished...")


if __name__ == '__main__':
    main()
