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
        tensor_batch_obs = isaac_drive_env.reset(batch_num=BATCH_NUM)
        while True:
            if True:  # agent
                tensor_batch_oneTime_action_xy = agent(tensor_batch_obs)  # [B, 2]
            else:  # random
                tensor_batch_oneTime_action_xy = torch.randn(BATCH_NUM, 2, device=DEVICE)  # [B, 2]
            reward, done = isaac_drive_env.step(tensor_batch_oneTime_action_xy)
            isaac_drive_env.render()
            if done:
                break

    print("Finished")


if __name__ == '__main__':
    main()
