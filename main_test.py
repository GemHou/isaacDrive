import time
import tqdm
import torch

from utils_agent import Agent
from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu
SCENE_NUM = 2
BATCH_NUM = 1
RENDER_FLAG = True


def prepare_agent(obs_dim):
    agent = Agent(obs_dim=obs_dim)
    state_dict = torch.load("data/interim/state_dict_grad.pt", map_location=DEVICE)
    agent.load_state_dict(state_dict)
    return agent


def sim_one_epoch(isaac_drive_env, agent, tensor_batch_obs):
    while True:
        if True:  # network
            tensor_batch_oneTime_action_xy = agent(tensor_batch_obs)  # [B, 2]
        else:  # rule
            # tensor_batch_oneTime_action_xy = torch.randn(BATCH_NUM, 2, device=DEVICE)  # [B, 2]
            # tensor_batch_oneTime_action_xy = torch.zeros(BATCH_NUM, 2, device=DEVICE)  # [B, 2]
            tensor_batch_oneTime_action_xy = torch.ones(BATCH_NUM, 2, device=DEVICE)  # [B, 2]
        reward, done, tensor_batch_obs = isaac_drive_env.step(tensor_batch_oneTime_action_xy)
        # print("reward: ", reward)
        if RENDER_FLAG:
            isaac_drive_env.render()
        if done:
            break


def main():
    start_time = time.time()

    # prepare environment
    isaac_drive_env = IsaacDriveEnv(device=DEVICE, scene_num=SCENE_NUM)

    # prepare agent
    agent = prepare_agent(obs_dim=isaac_drive_env.observation_space.shape[0])

    for _ in tqdm.tqdm(range(500)):
        tensor_batch_obs = isaac_drive_env.reset(batch_num=BATCH_NUM, mode="Train")  # Train Test
        sim_one_epoch(isaac_drive_env, agent, tensor_batch_obs)

    print("all time: ", time.time() - start_time)

    print("Finished")


if __name__ == '__main__':
    main()
