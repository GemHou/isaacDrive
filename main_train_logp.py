import tqdm
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal

from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu
NUM_EPOCH = 100
BATCH_NUM = 1


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def main():
    env = IsaacDriveEnv(device=DEVICE)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    mu_net = mlp([obs_dim] + [64, 64] + [act_dim], activation=nn.Tanh)
    v_net = mlp([obs_dim] + [64, 64] + [1], activation=nn.Tanh)
    log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
    log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    pi_lr = 3e-4
    vf_lr = 1e-3
    pi_optimizer = Adam([{'params': mu_net.parameters()},
                         {'params': log_std}
                         ],
                        lr=pi_lr)
    vf_optimizer = Adam(v_net.parameters(), lr=vf_lr)

    for epoch in tqdm.tqdm(range(NUM_EPOCH)):
        list_tensor_batch_obs = []
        list_tensor_batch_action_xy = []
        list_tensor_batch_reward = []
        list_tensor_batch_value = []
        list_tensor_batch_logp_a = []

        tensor_batch_obs = env.reset(batch_num=BATCH_NUM)
        while True:
            tensor_batch_mu = mu_net(tensor_batch_obs)  # [B, 2]
            std = torch.exp(log_std)
            pi = Normal(tensor_batch_mu, std)
            tensor_batch_action_xy = pi.sample()
            tensor_batch_logp_a = pi.log_prob(tensor_batch_action_xy).sum(axis=-1)
            tensor_batch_value = v_net(tensor_batch_obs)
            tensor_batch_reward, bool_done, tensor_batch_obs_next = env.step(tensor_batch_action_xy)

            list_tensor_batch_obs.append(tensor_batch_obs)
            list_tensor_batch_action_xy.append(tensor_batch_action_xy)
            list_tensor_batch_reward.append(tensor_batch_reward)
            list_tensor_batch_value.append(tensor_batch_value)
            list_tensor_batch_logp_a.append(tensor_batch_logp_a)

            tensor_batch_obs = tensor_batch_obs_next

            if bool_done:
                break

    print("Finished...")


if __name__ == '__main__':
    main()
