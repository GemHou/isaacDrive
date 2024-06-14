import time
import tqdm
import wandb
import torch
import torch.optim as optim
# from matplotlib import pyplot as plt

from utils_agent import Agent
from utils_isaac_drive_env import IsaacDriveEnv

torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cpu")  # cuda:0 cpu
RENDER_FLAG = True
SCENE_NUM = 2
BATCH_NUM = 1
RESUME_NAME = "20240614_5900X_grad_s2b1_obs198-7"
NUM_EPOCH = 50


def main():
    wandb.init(
        project="isaac_drive",
        resume=RESUME_NAME  # HjScenarioEnv
    )

    isaac_drive_env = IsaacDriveEnv(device=DEVICE, scene_num=SCENE_NUM)
    obs_dim = isaac_drive_env.observation_space.shape[0]
    agent = Agent(obs_dim=obs_dim)
    # state_dict = torch.load("./data/interim/state_dict_grad.pt", map_location=DEVICE)
    # agent.load_state_dict(state_dict)
    agent.to(DEVICE)
    lr = 0.001
    num_epoch = NUM_EPOCH
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    list_float_loss = []

    start_time = time.time()

    for _ in tqdm.tqdm(range(num_epoch)):

        tensor_batch_obs = isaac_drive_env.reset(batch_num=BATCH_NUM, mode="Train")
        optimizer.zero_grad()
        list_tensor_time_loss = []
        while True:
            # generate action
            if True:  # agent
                tensor_batch_action_xy = agent(tensor_batch_obs)  # [B, 2]
            else:  # random
                tensor_batch_action_xy = torch.randn(BATCH_NUM, 2, device=DEVICE)  # [B, 2]

            reward, done, tensor_batch_obs = isaac_drive_env.step(tensor_batch_action_xy)
            tensor_time_loss = - reward
            list_tensor_time_loss.append(tensor_time_loss)
            if done:
                break
        tensor_epoch_loss = torch.stack(list_tensor_time_loss, dim=1)
        loss_final = tensor_epoch_loss.mean(dim=-1).mean()
        reward_per_step = -loss_final.item()
        wandb.log({"reward_per_step": reward_per_step})
        list_float_loss.append(loss_final.item())
        loss_final.backward()
        optimizer.step()
    print("update network time: ", time.time() - start_time)  # 15 second
    if not RENDER_FLAG:
        total_frame = 100 * BATCH_NUM * 253
        print("throughput: ", total_frame / (time.time() - start_time))

    torch.save(agent.state_dict(), "./data/interim/state_dict_grad.pt")

    print("Finished...")


if __name__ == '__main__':
    main()
