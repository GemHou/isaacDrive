import torch
from torch import nn as nn


class Agent(nn.Module):
    def __init__(self, obs_dim):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)  # , dtype=torch.float
        self.fc2 = nn.Linear(64, 64)  # , dtype=torch.float
        self.fc3 = nn.Linear(64, 64)  # , dtype=torch.float
        self.fc4 = nn.Linear(64, 2)  # , dtype=torch.float
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = torch.tanh(x) * 5
        return x
