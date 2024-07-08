import time
import tqdm
import torch

from utils_agent import Agent, AgentAcceleration, AgentVehicleDynamic
from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu
SCENE_NUM = 100
BATCH_NUM = 90
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
    tensor_batch_obs = isaac_drive_env.reset(batch_num=BATCH_NUM, data_mode=TRAIN_TEST_MODE)  # Train Test
    list_tensor_time_reward = []
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
    tensor_epoch_reward = torch.stack(list_tensor_time_reward)
    reward_per_step = torch.mean(tensor_epoch_reward)
    # print("reward_per_step: ", reward_per_step)


def main():
    # prepare environment
    isaac_drive_env = IsaacDriveEnv(device=DEVICE, scene_num=SCENE_NUM, loop_mode=TEST_LOOP_MODE)

    # prepare agent
    # agent = prepare_agent()

    start_time = time.time()
    for _ in tqdm.tqdm(range(50)):
        state_dict = torch.load("data/interim/state_dict_grad.pt", map_location=DEVICE)
        # agent.load_state_dict(state_dict)
        sim_one_epoch(isaac_drive_env)

    simulation_time = time.time() - start_time
    print("simulation time: ", simulation_time)
    throught = BATCH_NUM * 254 / simulation_time
    print("throught: ", throught)

    print("Finished")


if __name__ == '__main__':
    main()
