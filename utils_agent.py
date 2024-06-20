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


class AgentAcceleration(nn.Module):
    def __init__(self):  # , obs_dim
        super(AgentAcceleration, self).__init__()
        self.fc_other_first = nn.Linear(198, 64)  # , dtype=torch.float
        self.fc_other_hid1 = nn.Linear(64, 64)  # , dtype=torch.float
        # self.fc_other_hid2 = nn.Linear(64, 64)  # , dtype=torch.float
        # self.fc_other_hid3 = nn.Linear(64, 64)  # , dtype=torch.float

        self.fc_ego_first = nn.Linear(202 - 198, 64)

        self.fc_last = nn.Linear(64 + 64, 2)  # , dtype=torch.float

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.friction = 1.0  #

    def forward(self, dict_tensor_batch_obs):
        tensor_batch_obs_other = dict_tensor_batch_obs["tensor_batch_obs_other"]
        tensor_batch_ego = dict_tensor_batch_obs["tensor_batch_ego"]
        bs, _ = tensor_batch_ego.size()
        tensor_batch_speed = tensor_batch_ego[:, 0]
        tensor_batch_yaw = tensor_batch_ego[:, 1]

        x_other = self.fc_other_first(tensor_batch_obs_other)
        x_other = self.tanh(x_other)
        x_other = self.fc_other_hid1(x_other)
        x_other = self.tanh(x_other)
        # x_other = self.fc_other_hid2(x_other)
        # x_other = self.tanh(x_other)
        # x_other = self.fc_other_hid3(x_other)
        # x_other = self.tanh(x_other)

        x_ego = self.fc_ego_first(tensor_batch_ego)  # [B, 64]

        x = torch.cat((x_other, x_ego), dim=-1)

        x = self.fc_last(x)  # [B, 2]
        x = self.tanh(x)  # [B, 2] -1~1
        action_accelerationXy = x * 9.81 * self.friction  # -10 ～ 10

        tensor_batch_speedX = tensor_batch_speed * torch.cos(tensor_batch_yaw)
        tensor_batch_speedY = tensor_batch_speed * torch.sin(tensor_batch_yaw)

        tensor_batch_speedX_new = tensor_batch_speedX + action_accelerationXy[:, 0] * 0.1
        tensor_batch_speedY_new = tensor_batch_speedY + action_accelerationXy[:, 1] * 0.1

        tensor_batch_deltaPosX_new = tensor_batch_speedX_new * 0.1
        tensor_batch_deltaPosY_new = tensor_batch_speedY_new * 0.1

        action_deltaPosXy = torch.stack([tensor_batch_deltaPosX_new, tensor_batch_deltaPosY_new], dim=1)

        # action_deltaPosXy = None  # [B, 2]
        return action_deltaPosXy


class AgentVehicleDynamic(nn.Module):
    def __init__(self):
        super(AgentVehicleDynamic, self).__init__()

        self.fc_other_first = nn.Linear(198, 64)  # , dtype=torch.float
        self.fc_other_hid1 = nn.Linear(64, 64)  # , dtype=torch.float
        self.fc_other_hid2 = nn.Linear(64, 64)  # , dtype=torch.float
        self.fc_other_hid3 = nn.Linear(64, 64)  # , dtype=torch.float

        self.fc_ego_first = nn.Linear(202-198, 64)

        self.fc_last = nn.Linear(64 + 64, 2)  # , dtype=torch.float

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.turning_radis_mech = 2.5  # 5
        self.friction = 2.0  # 0.9

    def calc_vehicle_dynamic(self, action_throttleWheel, tensor_batch_speed, tensor_batch_yaw):
        tensor_batch_acceleration = action_throttleWheel[:, 0] * 9.81 * self.friction
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

    def forward(self, dict_tensor_batch_obs):
        """
            x: [B, 200]
        """
        tensor_batch_obs_other = dict_tensor_batch_obs["tensor_batch_obs_other"]
        tensor_batch_ego = dict_tensor_batch_obs["tensor_batch_ego"]
        bs, _ = tensor_batch_ego.size()
        tensor_batch_speed = tensor_batch_ego[:, 0]
        tensor_batch_yaw = tensor_batch_ego[:, 1]

        x_other = self.fc_other_first(tensor_batch_obs_other)
        x_other = self.tanh(x_other)
        x_other = self.fc_other_hid1(x_other)
        x_other = self.tanh(x_other)
        x_other = self.fc_other_hid2(x_other)
        x_other = self.tanh(x_other)
        x_other = self.fc_other_hid3(x_other)
        x_other = self.tanh(x_other)

        x_ego = self.fc_ego_first(tensor_batch_ego)  # [B, 64]

        x = torch.cat((x_other, x_ego), dim=-1)
        x = self.fc_last(x)
        action_throttleWheel = torch.tanh(x)
        # action_throttleWheel = torch.zeros(bs, 2, device=action_throttleWheel.device)  # [B, 2]
        # action_throttleWheel[:, 0] = torch.ones(bs, device=action_throttleWheel.device) * 1  # throttle debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # action_throttleWheel[:, 1] = torch.ones(bs, device=action_throttleWheel.device) * 0.00000001  # wheel debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        action_deltaPosXy = self.calc_vehicle_dynamic(action_throttleWheel, tensor_batch_speed, tensor_batch_yaw)
        # action_deltaPosXy = action_throttleWheel * 5

        # print("action_throttleWheel: ", action_throttleWheel, "action_deltaPosXy: ", action_deltaPosXy)

        return action_deltaPosXy  # [B, 2]
