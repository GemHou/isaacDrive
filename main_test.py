import torch

from utils_agent import Agent
from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu
BATCH_NUM = 1


def main():
    # prepare agent
    agent = Agent()
    state_dict = torch.load("./data/interim/state_dict_temp.pt", map_location=DEVICE)
    agent.load_state_dict(state_dict)

    # prepare environment
    isaac_drive_env = IsaacDriveEnv(device=DEVICE)

    for _ in range(10):
        isaac_drive_env.reset(batch_num=BATCH_NUM)

    print("Finished")


if __name__ == '__main__':
    main()
