import time
import tqdm
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from utils_agent import Agent
from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu
RENDER_FLAG = True
BATCH_NUM = 1


def main():
    isaac_drive_env = IsaacDriveEnv(device=DEVICE)
    agent = Agent()
    agent.to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=0.0001)

    list_float_loss = []

    start_time = time.time()

    for epoch in tqdm.tqdm(range(100)):

        tensor_batch_obs = isaac_drive_env.reset(batch_num=BATCH_NUM)
        # optimizer.zero_grad()
        # list_tensor_loss = []
        while True:
            optimizer.zero_grad()
            # generate action
            if True:  # agent
                tensor_batch_action_xy = agent(tensor_batch_obs)  # [B, 2]
            else:  # random
                tensor_batch_action_xy = torch.randn(BATCH_NUM, 2, device=DEVICE)  # [B, 2]

            reward, done, tensor_batch_obs = isaac_drive_env.step(tensor_batch_action_xy)
            loss = - reward
            # list_tensor_loss.append(loss)
            loss_mean = loss.mean()
            loss_mean.backward()
            optimizer.step()
            list_float_loss.append(loss_mean.item())
            if RENDER_FLAG and len(list_float_loss) % 100 == 0:
                plt.cla()
                plt.plot(list_float_loss)
                plt.pause(0.05)
            if done:
                break
        # loss_epoch = torch.stack(list_tensor_loss)
        # loss_sum = loss_epoch.mean()
        # print("loss_sum: ", loss_sum)
        # list_float_loss.append(loss_sum.item())
        # if RENDER_FLAG and len(list_float_loss) % 10 == 0:
        #     plt.cla()
        #     plt.plot(list_float_loss)
        #     plt.pause(0.05)
        # loss_sum.backward()
        # optimizer.step()
    print("update network time: ", time.time() - start_time)  # 15 second

    torch.save(agent.state_dict(), "./data/interim/state_dict_temp.pt")

    plt.cla()
    plt.plot(list_float_loss)
    plt.show()

    print("Finished...")


if __name__ == '__main__':
    main()
