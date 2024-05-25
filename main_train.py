import time
import tqdm
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from utils_agent import Agent
from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu
RENDER_FLAG = False
BATCH_NUM = 9


def main():
    isaac_drive_env = IsaacDriveEnv(device=DEVICE)
    agent = Agent()
    agent.to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)

    list_loss = []

    start_time = time.time()

    for epoch in tqdm.tqdm(range(2000)):
        optimizer.zero_grad()

        tensor_batch_obs = isaac_drive_env.reset(batch_num=BATCH_NUM)

        # generate action
        if True:  # agent
            tensor_batch_action_xy = agent(tensor_batch_obs)  # [20, 254, 2]
        else:  # random
            tensor_batch_action_xy = torch.randn(BAG_NUM, 254, 2, device=DEVICE)  # [20, 254, 2]

        reward = isaac_drive_env.step(tensor_batch_action_xy)

        loss = - reward
        loss_sum = loss.sum()
        # print("loss_sum: ", loss_sum)
        list_loss.append(loss_sum.item())
        if RENDER_FLAG and epoch % 10 == 0:
            # isaac_drive_env.render()
            plt.cla()
            plt.plot(list_loss)
            plt.pause(0.05)

        loss_sum.backward()
        optimizer.step()
    print("update network time: ", time.time() - start_time)  # 15 second

    torch.save(agent.state_dict(), "./data/interim/state_dict_temp.pt")

    plt.cla()
    plt.plot(list_loss)
    plt.show()

    print("Finished...")


if __name__ == '__main__':
    main()
