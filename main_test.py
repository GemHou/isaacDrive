import time
import tqdm
import torch

from utils_agent import Agent
from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu
BATCH_NUM = 2
RENDER_FLAG = True


def prepare_agent():
    agent = Agent()
    state_dict = torch.load("./data/interim/state_dict_temp.pt", map_location=DEVICE)
    agent.load_state_dict(state_dict)


def sim_one_epoch(isaac_drive_env):
    while True:
        if False:  # agent
            tensor_batch_oneTime_action_xy = agent(tensor_batch_obs)  # [B, 2]
        else:  # random
            # tensor_batch_oneTime_action_xy = torch.randn(BATCH_NUM, 2, device=DEVICE)  # [B, 2]
            tensor_batch_oneTime_action_xy = torch.zeros(BATCH_NUM, 2, device=DEVICE)  # [B, 2]
        reward, done, tensor_batch_obs = isaac_drive_env.step(tensor_batch_oneTime_action_xy)
        # print("reward: ", reward)
        if RENDER_FLAG:
            isaac_drive_env.render()
        if done:
            break


def main():
    start_time = time.time()

    # prepare agent
    prepare_agent()

    # prepare environment
    isaac_drive_env = IsaacDriveEnv(device=DEVICE)

    for _ in tqdm.tqdm(range(500)):
        tensor_batch_obs = isaac_drive_env.reset(batch_num=BATCH_NUM)
        sim_one_epoch(isaac_drive_env)

    print("all time: ", time.time() - start_time)

    print("Finished")


if __name__ == '__main__':
    main()
