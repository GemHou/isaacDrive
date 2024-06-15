import torch
from torch import nn as nn


class Agent(nn.Module):
    def __init__(self, obs_dim):
        super(Agent, self).__init__()
        self.fc_first = nn.Linear(obs_dim, 64)  # , dtype=torch.float
        self.fc_hid1 = nn.Linear(64, 64)  # , dtype=torch.float
        # self.fc_hid2 = nn.Linear(64, 64)  # , dtype=torch.float
        # self.fc_hid3 = nn.Linear(64, 64)  # , dtype=torch.float
        self.fc_last = nn.Linear(64, 2)  # , dtype=torch.float
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc_first(x)
        x = self.tanh(x)
        x = self.fc_hid1(x)
        x = self.tanh(x)
        # x = self.fc_hid2(x)
        # x = self.tanh(x)
        # x = self.fc_hid3(x)
        # x = self.tanh(x)
        x = self.fc_last(x)
        x = torch.tanh(x) * 5
        return x
