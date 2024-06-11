import time
import tqdm
import wandb
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from utils_agent import Agent
from utils_isaac_drive_env import IsaacDriveEnv

torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cpu")  # cuda:0 cpu
RENDER_FLAG = True
BATCH_NUM = 100
BACKWARD_FREQ = "Epoch"  # "Epoch"  "Step"
RESUME_NAME = "grad_s100b100_20240611"


def main():
    wandb.init(
        project="isaac_drive",
        resume=RESUME_NAME  # HjScenarioEnv
    )

    isaac_drive_env = IsaacDriveEnv(device=DEVICE)
    obs_dim = isaac_drive_env.observation_space.shape[0]
    agent = Agent(obs_dim=obs_dim)
    # state_dict = torch.load("./data/interim/state_dict_grad.pt", map_location=DEVICE)
    # agent.load_state_dict(state_dict)
    agent.to(DEVICE)
    if BACKWARD_FREQ == "Epoch":
        lr = 0.001
        num_epoch = 100
    elif BACKWARD_FREQ == "Step":
        lr = 0.00001
        num_epoch = 50
    else:
        raise
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    list_float_loss = []

    start_time = time.time()

    for epoch in tqdm.tqdm(range(num_epoch)):

        tensor_batch_obs = isaac_drive_env.reset(batch_num=BATCH_NUM)
        if BACKWARD_FREQ == "Epoch":
            optimizer.zero_grad()
            list_tensor_time_loss = []
        while True:
            if BACKWARD_FREQ == "Step":
                optimizer.zero_grad()
            # generate action
            if True:  # agent
                tensor_batch_action_xy = agent(tensor_batch_obs)  # [B, 2]
            else:  # random
                tensor_batch_action_xy = torch.randn(BATCH_NUM, 2, device=DEVICE)  # [B, 2]

            reward, done, tensor_batch_obs = isaac_drive_env.step(tensor_batch_action_xy)
            tensor_time_loss = - reward
            if BACKWARD_FREQ == "Step":
                loss_mean = tensor_time_loss.mean()
                print("loss_mean: ", loss_mean)
                loss_mean.backward(retain_graph=True)
                optimizer.step()
                list_float_loss.append(loss_mean.item())
                if RENDER_FLAG and len(list_float_loss) % 100 == 0:
                    plt.cla()
                    plt.plot(list_float_loss)
                    plt.pause(0.05)
            if BACKWARD_FREQ == "Epoch":
                list_tensor_time_loss.append(tensor_time_loss)
            if done:
                break
        if BACKWARD_FREQ == "Epoch":
            tensor_epoch_loss = torch.stack(list_tensor_time_loss, dim=1)
            loss_final = tensor_epoch_loss.sum(dim=-1).mean()
            return_epoch = -loss_final.item()
            wandb.log({"return_epoch": return_epoch})
            list_float_loss.append(loss_final.item())
            if RENDER_FLAG and len(list_float_loss) % 10 == 0:
                plt.cla()
                plt.plot(list_float_loss)
                plt.pause(0.05)
            loss_final.backward()
            optimizer.step()
    print("update network time: ", time.time() - start_time)  # 15 second
    if not RENDER_FLAG:
        total_frame = 100 * BATCH_NUM * 253
        print("throughput: ", total_frame / (time.time() - start_time))

    torch.save(agent.state_dict(), "./data/interim/state_dict_grad.pt")

    plt.cla()
    plt.plot(list_float_loss)
    plt.show()

    print("Finished...")


if __name__ == '__main__':
    main()
