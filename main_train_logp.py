import torch
import torch.nn as nn
from torch.optim import Adam

from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu


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

    pi_lr = 3e-4
    vf_lr = 1e-3
    pi_optimizer = Adam(mu_net.parameters(), lr=pi_lr)
    vf_optimizer = Adam(v_net.parameters(), lr=vf_lr)

    print("Finished...")


if __name__ == '__main__':
    main()
