import math
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
        action_deltaPosXy = torch.tanh(x) * 5  # [B, 2]
        return action_deltaPosXy


class AgentVehicleDynamic(nn.Module):
    def __init__(self, obs_dim):
        super(AgentVehicleDynamic, self).__init__()
        self.fc_first = nn.Linear(obs_dim, 64)  # , dtype=torch.float
        self.fc_hid1 = nn.Linear(64, 64)  # , dtype=torch.float
        # self.fc_hid2 = nn.Linear(64, 64)  # , dtype=torch.float
        # self.fc_hid3 = nn.Linear(64, 64)  # , dtype=torch.float
        self.fc_last = nn.Linear(64 + 64, 2)  # , dtype=torch.float
        self.fc_2_1 = nn.Linear(210-198, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.turning_radis_mech = 30

    def calc_vehicle_dynamic(self, action_throttleWheel, tensor_batch_speed, tensor_batch_yaw):
        tensor_batch_acceleration = action_throttleWheel[:, 0] * 9.81 * 0.3
        tensor_batch_speed_new = tensor_batch_speed + tensor_batch_acceleration * 0.1
        tensor_batch_speed_new = torch.max(tensor_batch_speed_new, torch.ones_like(tensor_batch_speed_new) * 0.01)

        tensor_batch_wheel = action_throttleWheel[:, 1]  # [B]
        tensor_batch_turning_radis = self.turning_radis_mech / - tensor_batch_wheel  # [B]
        tensor_batch_deltaYaw = tensor_batch_speed_new * 0.1 / tensor_batch_turning_radis

        tensor_batch_yaw_new = tensor_batch_yaw + tensor_batch_deltaYaw

        action_deltaPosX = torch.cos(tensor_batch_yaw_new) * tensor_batch_speed_new * 0.1  # [B]
        action_deltaPosY = torch.sin(tensor_batch_yaw_new) * tensor_batch_speed_new * 0.1  # [B]
        action_deltaPosXy = torch.stack([action_deltaPosX, action_deltaPosY], dim=1)
        return action_deltaPosXy

    def forward(self, x):
        """
            x: [B, 200]
        """
        assert x.size(1) == 210
        bs, _ = x.size()
        tensor_batch_speed = x[:, 0]
        tensor_batch_yaw = x[:, 1]

        x1 = self.fc_first(x)
        x1 = self.tanh(x1)
        x1 = self.fc_hid1(x1)
        x1 = self.tanh(x1)
        # x1 = self.fc_hid2(x1)
        # x1 = self.tanh(x1)
        # x1 = self.fc_hid3(x1)
        # x1 = self.tanh(x1)

        x2 = self.fc_2_1(x[:, 0:210-198])  # [B, 64]

        x = torch.cat((x1, x2), dim=-1)
        x = self.fc_last(x)
        action_throttleWheel = torch.tanh(x)
        # action_throttleWheel = torch.zeros(bs, 2, device=action_throttleWheel.device)  # [B, 2]
        # action_throttleWheel[:, 0] = torch.ones(bs, device=action_throttleWheel.device) * 1  # throttle debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # action_throttleWheel[:, 1] = torch.ones(bs, device=action_throttleWheel.device) * 0.00000001  # wheel debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        action_deltaPosXy = self.calc_vehicle_dynamic(action_throttleWheel, tensor_batch_speed, tensor_batch_yaw)
        # action_deltaPosXy = action_throttleWheel * 5

        # print("action_throttleWheel: ", action_throttleWheel, "action_deltaPosXy: ", action_deltaPosXy)

        return action_deltaPosXy  # [B, 2]
