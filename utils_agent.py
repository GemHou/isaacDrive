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

        self.encoder_other = "FC"  # FC Pool  # both is okay, but Pool is slower and worse
        self.decoder_method = "Speed"  # Speed Acceleration Vehicle

        if self.encoder_other == "FC":
            self.fc_other_first = nn.Linear(50*4, 64)  # 13k
            self.fc_other_hid1 = nn.Linear(64, 64)  # 4k
        elif self.encoder_other == "Pool":
            self.fc_other_first = nn.Linear(4, 64)
            self.fc_other_hid1 = nn.Linear(64, 64)  # 4k
            self.fc_other_hid2 = nn.Linear(64, 64)  # 4k
            self.fc_other_hid3 = nn.Linear(64, 64)  # 4k
        else:
            raise

        self.fc_ego_first = nn.Linear(4, 64)
        self.fc_ego_hid1 = nn.Linear(64, 64)

        # self.fc_cheat_first = nn.Linear(2, 64)
        # self.fc_cheat_hid1 = nn.Linear(64, 64)


        self.decoder_fc_first = nn.Linear(64 + 64, 64)
        self.decoder_fc_last = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.friction = 1.0  #

        # self.zero_params = nn.Parameter(torch.tensor(999.0))

    def decode(self, tensor_batch_speed, tensor_batch_yaw, x_ego, x_other):
        if self.decoder_method == "Acceleration":
            x = torch.cat((x_ego, x_other), dim=-1)  # , x_cheat
            x = self.decoder_fc_first(x)  # [B, 64]
            x = self.tanh(x)
            x = self.decoder_fc_last(x)  # [B, 2]
            x = self.tanh(x)  # [B, 2] -1~1
            action_accelerationXy = x * 9.81 * self.friction  # -10 ～ 10
            tensor_batch_speedX = tensor_batch_speed * torch.cos(tensor_batch_yaw)
            tensor_batch_speedY = tensor_batch_speed * torch.sin(tensor_batch_yaw)
            tensor_batch_speedX_new = tensor_batch_speedX + action_accelerationXy[:, 0] * 0.1
            tensor_batch_speedY_new = tensor_batch_speedY + action_accelerationXy[:, 1] * 0.1
            tensor_batch_deltaPosX_new = tensor_batch_speedX_new * 0.1
            tensor_batch_deltaPosY_new = tensor_batch_speedY_new * 0.1
            action_deltaPosXy = torch.stack([tensor_batch_deltaPosX_new, tensor_batch_deltaPosY_new], dim=1)
        elif self.decoder_method == "Speed":
            x = torch.cat((x_ego, x_other), dim=-1)  # , x_cheat
            x = self.decoder_fc_first(x)  # [B, 64]
            x = self.tanh(x)
            x = self.decoder_fc_last(x)  # [B, 2]
            x = self.tanh(x)  # [B, 2] -1~1
            tensor_batch_speedX_new = x[:, 0] * 10
            tensor_batch_speedY_new = x[:, 1] * 10
            tensor_batch_deltaPosX_new = tensor_batch_speedX_new * 0.1
            tensor_batch_deltaPosY_new = tensor_batch_speedY_new * 0.1
            action_deltaPosXy = torch.stack([tensor_batch_deltaPosX_new, tensor_batch_deltaPosY_new], dim=1)
        else:
            raise
        return action_deltaPosXy

    def forward(self, dict_tensor_batch_obs):
        tensor_batch_obs_other = dict_tensor_batch_obs["tensor_batch_obs_other"]  # [B, 99, 2]
        tensor_batch_ego = dict_tensor_batch_obs["tensor_batch_ego"]  # [B, 4]
        tensor_batch_cheat = dict_tensor_batch_obs["tensor_batch_cheat"]  # [B, 2]
        bs, _ = tensor_batch_ego.size()
        tensor_batch_speed = tensor_batch_ego[:, 0]
        tensor_batch_yaw = tensor_batch_ego[:, 1]

        # tensor_batch_obs_other = torch.where(tensor_batch_obs_other == 9999.0,
        #                                      self.zero_params.expand_as(tensor_batch_obs_other),
        #                                      tensor_batch_obs_other)

        if self.encoder_other == "FC":
            tensor_batch_obs_other_flat = tensor_batch_obs_other.reshape(-1, 50 * 4)
            x_other = self.fc_other_first(tensor_batch_obs_other_flat)  # [B, 64]
            x_other = self.tanh(x_other)
            x_other = self.fc_other_hid1(x_other)  # [B, 64]
        elif self.encoder_other == "Pool":
            x_other = self.fc_other_first(tensor_batch_obs_other)  # [B, 99, 64]
            x_other = self.tanh(x_other)
            x_other = self.fc_other_hid1(x_other)  # [B, 99, 64]
            x_other = x_other.transpose(1, 2)  # [B, 64, 99]
            x_other = self.pool(x_other)  # [B, 64, 1]
            x_other = x_other.squeeze(-1)  # [B, 64]
            x_other = self.fc_other_hid2(x_other)  # [B, 64]
            x_other = self.tanh(x_other)
            x_other = self.fc_other_hid3(x_other)  # [B, 64]
        else:
            raise

        x_ego = self.fc_ego_first(tensor_batch_ego)  # [B, 64]
        x_ego = self.tanh(x_ego)
        x_ego = self.fc_ego_hid1(x_ego)  # [B, 64]

        # x_cheat = self.fc_cheat_first(tensor_batch_cheat)  # [B, 64]
        # x_cheat = self.tanh(x_cheat)
        # x_cheat = self.fc_cheat_hid1(x_cheat)  # [B, 64]

        action_deltaPosXy = self.decode(tensor_batch_speed, tensor_batch_yaw, x_ego, x_other)

        # action_deltaPosXy = None  # [B, 2]
        return action_deltaPosXy


class AgentVehicleDynamic(nn.Module):
    def __init__(self):
        super(AgentVehicleDynamic, self).__init__()

        self.fc_other_first = nn.Linear(198, 64)  # , dtype=torch.float
        self.fc_other_hid1 = nn.Linear(64, 64)  # , dtype=torch.float
        self.fc_other_hid2 = nn.Linear(64, 64)  # , dtype=torch.float
        self.fc_other_hid3 = nn.Linear(64, 64)  # , dtype=torch.float

        self.fc_ego_first = nn.Linear(4, 64)

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
