import torch
from torch import nn as nn


class Agent(nn.Module):
    def __init__(self, obs_dim):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)  # , dtype=torch.float
        self.fc2 = nn.Linear(64, 2)  # , dtype=torch.float
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
        x4 = torch.tanh(x3)  # 0.001
        return x4
