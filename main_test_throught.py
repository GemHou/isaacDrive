import time
import tqdm
import torch
import numpy as np

from utils_agent import Agent, AgentAcceleration, AgentVehicleDynamic
from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu
SCENE_NUM = 184  # 184 185
BATCH_NUM = 128  # int(SCENE_NUM*0.9)
RENDER_FLAG = False  # True False
TRAIN_TEST_MODE = "Train"  # Train Test
TEST_LOOP_MODE = "Closed"  # Closed Open


def prepare_agent():
    # agent = Agent(obs_dim=obs_dim)
    agent = AgentAcceleration()
    # agent = AgentVehicleDynamic(obs_dim=obs_dim)
    state_dict = torch.load("data/interim/state_dict_grad.pt", map_location=DEVICE)
    agent.load_state_dict(state_dict)
    return agent


def sim_one_epoch(isaac_drive_env):
    print("BATCH_NUM: ", BATCH_NUM)
    tensor_batch_obs = isaac_drive_env.reset(batch_num=BATCH_NUM, data_mode=TRAIN_TEST_MODE)  # Train Test
    list_tensor_time_reward = []
    start_time = time.time()
    while True:
        # tensor_batch_oneTime_action_xy = torch.randn(BATCH_NUM, 2, device=DEVICE)  # [B, 2]
        # tensor_batch_oneTime_action_xy = torch.zeros(BATCH_NUM, 2, device=DEVICE)  # [B, 2]
        tensor_batch_oneTime_action_xy = torch.ones(BATCH_NUM, 2, device=DEVICE)  # [B, 2]

        tensor_time_reward, done, tensor_batch_obs, info = isaac_drive_env.step(tensor_batch_oneTime_action_xy)
        list_tensor_time_reward.append(tensor_time_reward)
        if RENDER_FLAG:
            isaac_drive_env.render()
        if done:
            break
    throughout = BATCH_NUM * 254 / (time.time() - start_time)
    print("")
    tensor_epoch_reward = torch.stack(list_tensor_time_reward)
    reward_per_step = torch.mean(tensor_epoch_reward)
    # print("reward_per_step: ", reward_per_step)
    return throughout


def main():
    # prepare environment
    isaac_drive_env = IsaacDriveEnv(device=DEVICE, scene_num=SCENE_NUM, loop_mode=TEST_LOOP_MODE)

    # prepare agent
    # agent = prepare_agent()

    list_throughout = []

    for _ in tqdm.tqdm(range(50)):
        # state_dict = torch.load("data/interim/state_dict_grad.pt", map_location=DEVICE)
        # agent.load_state_dict(state_dict)
        throughout = sim_one_epoch(isaac_drive_env)
        list_throughout.append(throughout)

    print("np.mean(list_throughout): ", np.mean(list_throughout))
    print("np.std(list_throughout): ", np.std(list_throughout))

    print("Finished")


if __name__ == '__main__':
    main()
