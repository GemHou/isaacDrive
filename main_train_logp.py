import torch

from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu


def main():
    env = IsaacDriveEnv(device=DEVICE)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    print("Finished...")


if __name__ == '__main__':
    main()
