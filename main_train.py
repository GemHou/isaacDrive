import time
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from utils_agent import Agent
from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu


def main():
    isaac_drive_env = IsaacDriveEnv(device=DEVICE)
    agent = Agent()
    agent.to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)

    list_loss = []

    start_time = time.time()

    for epoch in range(10000):
        optimizer.zero_grad()

        tensor_batch_obs = isaac_drive_env.reset()

        # generate action
        if True:  # agent
            tensor_batch_action_xy = agent(tensor_batch_obs)  # [20, 254, 2]
        else:  # random
            tensor_batch_action_xy = torch.randn(BAG_NUM, 254, 2, device=DEVICE)  # [20, 254, 2]

        tensor_batch_dis_start_withAction = isaac_drive_env.step(tensor_batch_action_xy)

        loss = - tensor_batch_dis_start_withAction
        loss_sum = loss.sum()
        print("loss_sum: ", loss_sum)
        list_loss.append(loss_sum.item())
        if epoch % 10 == 0:
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
