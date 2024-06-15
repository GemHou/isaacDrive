import time
import tqdm
import wandb
import torch
import torch.optim as optim
# from matplotlib import pyplot as plt

from utils_agent import Agent, AgentVehicleDynamic
from utils_isaac_drive_env import IsaacDriveEnv

torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cpu")  # cuda:0 cpu
RENDER_FLAG = True
SCENE_NUM = 100
TRAIN_BATCH_NUM = 90
TEST_BATCH_NUM = 10
RESUME_NAME = "20240615_5700U_grad_s100b90_obs200_r0802_layer3_2"
NUM_EPOCH = 20


def epoch_train(agent, isaac_drive_env, optimizer):
    tensor_batch_obs = isaac_drive_env.reset(batch_num=TRAIN_BATCH_NUM, mode="Train")
    optimizer.zero_grad()
    list_tensor_time_loss = []
    list_tensor_time_reward_gt = []
    list_tensor_time_reward_safe = []
    while True:
        # generate action
        if True:  # agent
            tensor_batch_action_xy = agent(tensor_batch_obs)  # [B, 2]
        else:  # random
            tensor_batch_action_xy = torch.randn(TRAIN_BATCH_NUM, 2, device=DEVICE)  # [B, 2]

        reward, done, tensor_batch_obs, info = isaac_drive_env.step(tensor_batch_action_xy)
        tensor_time_loss = - reward
        list_tensor_time_loss.append(tensor_time_loss)
        list_tensor_time_reward_gt.append(info["reward_gt_norm"])
        list_tensor_time_reward_safe.append(info["reward_safe_norm"])
        if done:
            break
    tensor_epoch_loss = torch.stack(list_tensor_time_loss, dim=1)
    tensor_epoch_reward_gt = torch.stack(list_tensor_time_reward_gt, dim=1)
    tensor_epoch_reward_safe = torch.stack(list_tensor_time_reward_safe, dim=1)
    loss_final = tensor_epoch_loss.mean(dim=-1).mean()
    reward_per_step = -loss_final.item()
    reward_gt = tensor_epoch_reward_gt.mean()
    reward_safe = tensor_epoch_reward_safe.mean()
    wandb.log({"train/reward_per_step": reward_per_step})
    wandb.log({"train/reward_per_step_gt": reward_gt})
    wandb.log({"train/reward_per_step_safe": reward_safe})
    loss_final.backward()
    optimizer.step()


def epoch_test(agent, isaac_drive_env):
    tensor_batch_obs = isaac_drive_env.reset(batch_num=TEST_BATCH_NUM, mode="Test")
    list_tensor_time_loss = []
    list_tensor_time_reward_gt = []
    list_tensor_time_reward_safe = []
    while True:
        tensor_batch_action_xy = agent(tensor_batch_obs)  # [B, 2]
        reward, done, tensor_batch_obs, info = isaac_drive_env.step(tensor_batch_action_xy)
        tensor_time_loss = - reward
        list_tensor_time_loss.append(tensor_time_loss)
        list_tensor_time_reward_gt.append(info["reward_gt_norm"])
        list_tensor_time_reward_safe.append(info["reward_safe_norm"])
        if done:
            break
    tensor_epoch_loss = torch.stack(list_tensor_time_loss, dim=1)
    tensor_epoch_reward_gt = torch.stack(list_tensor_time_reward_gt, dim=1)
    tensor_epoch_reward_safe = torch.stack(list_tensor_time_reward_safe, dim=1)
    loss_final = tensor_epoch_loss.mean(dim=-1).mean()
    reward_per_step = -loss_final.item()
    reward_gt = tensor_epoch_reward_gt.mean()
    reward_safe = tensor_epoch_reward_safe.mean()
    wandb.log({"test/reward_per_step": reward_per_step})
    wandb.log({"test/reward_per_step_gt": reward_gt})
    wandb.log({"test/reward_per_step_safe": reward_safe})


def main():
    wandb.init(
        project="isaac_drive",
        resume=RESUME_NAME  # HjScenarioEnv
    )

    isaac_drive_env = IsaacDriveEnv(device=DEVICE, scene_num=SCENE_NUM)
    obs_dim = isaac_drive_env.observation_space.shape[0]
    # agent = Agent(obs_dim=obs_dim)
    agent = AgentVehicleDynamic(obs_dim=obs_dim)
    # state_dict = torch.load("./data/interim/state_dict_grad.pt", map_location=DEVICE)
    # agent.load_state_dict(state_dict)
    agent.to(DEVICE)
    lr = 0.001
    num_epoch = NUM_EPOCH
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    start_time = time.time()

    for _ in tqdm.tqdm(range(num_epoch)):
        epoch_train(agent, isaac_drive_env, optimizer)
        epoch_test(agent, isaac_drive_env)

    print("update network time: ", time.time() - start_time)  # 15 second
    if not RENDER_FLAG:
        total_frame = 100 * TRAIN_BATCH_NUM * 253
        print("throughput: ", total_frame / (time.time() - start_time))

    torch.save(agent.state_dict(), "./data/interim/state_dict_grad.pt")

    print("Finished...")


if __name__ == '__main__':
    main()
