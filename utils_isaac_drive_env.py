import os
import time

import gym
import torch
import random
import numpy as np
from matplotlib import pyplot as plt

W_gt = 1.0
W_safe = 0


def get_file_names(path):
    file_names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_names.append(os.path.join(path, file))  # root
    return file_names


class IsaacDriveEnv:
    def get_fileName(self, scene_num):
        if False:
            list_str_path = [
                "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
                "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
                "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
                "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
                "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
                "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
                "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
                "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
                "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
                "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
            ]
        else:
            list_str_path = get_file_names("data/raw/data_left_100/")
            list_str_path = list_str_path * 10
        assert scene_num <= len(list_str_path)
        list_str_path = list_str_path[:scene_num]
        print("len(list_str_path): ", len(list_str_path))
        try:
            assert len(list_str_path) > 0
        except AssertionError:
            print("Warning!!!!!! you do not have the dataset...")
            print("Warning!!!!!! please change line 24 False to True to use the template dataset or contact gem hou...")
            raise
        return list_str_path

    def trans_fileName_to_npz(self, list_str_path):
        start_time = time.time()

        self.all_bag_num = len(list_str_path)
        list_npz_data = []
        for str_path in list_str_path:
            # print("str_path: ", str_path)
            npz_data = np.load(str_path, allow_pickle=True)
            list_npz_data.append(npz_data)
        print("file name 2 npz time per bag (ms): ", (time.time() - start_time) * 1000 / self.all_bag_num)
        return list_npz_data

    def trans_npz_to_tensor(self, list_npz_data):
        start_time = time.time()
        tensor_all_vectornet_object_feature = torch.zeros(self.all_bag_num, 254, 100, 16, 11).to(self.device)
        tensor_all_vectornet_object_mask = torch.zeros(self.all_bag_num, 254, 100, 16).to(self.device)
        tensor_all_vectornet_static_feature = torch.zeros(self.all_bag_num, 254, 80, 16, 6).to(self.device)
        tensor_all_ego_gt_traj = torch.zeros(self.all_bag_num, 254, 20, 2).to(self.device)
        tensor_all_ego_gt_traj_hist = torch.zeros(self.all_bag_num, 254, 10, 2).to(self.device)
        tensor_all_ego_gt_traj_long = torch.zeros(self.all_bag_num, 254, 60, 2).to(self.device)
        for npz_i in range(len(list_npz_data)):
            npz_data = list_npz_data[npz_i]  # 1ms

            numpy_vectornet_object_feature = npz_data['vectornet_object_feature']  # 32ms
            tensor_vectornet_object_feature = torch.from_numpy(numpy_vectornet_object_feature).to(self.device)  # 1ms
            npz_timestep = tensor_vectornet_object_feature.size(0)
            tensor_all_vectornet_object_feature[npz_i, :npz_timestep] = tensor_vectornet_object_feature  # 1ms
            tensor_all_vectornet_object_feature[npz_i, :npz_timestep] = tensor_vectornet_object_feature  # 1ms

            numpy_vectornet_object_mask = npz_data['vectornet_object_mask']  # 3ms
            tensor_vectornet_object_mask = torch.from_numpy(numpy_vectornet_object_mask).to(self.device)  # 3ms
            tensor_all_vectornet_object_mask[npz_i, :npz_timestep] = tensor_vectornet_object_mask  # 1ms

            numpy_vectornet_static_feature = npz_data['vectornet_static_feature']  # 14ms
            tensor_vectornet_static_feature = torch.from_numpy(numpy_vectornet_static_feature).to(self.device)  # 5ms
            tensor_all_vectornet_static_feature[npz_i, :npz_timestep] = tensor_vectornet_static_feature  # 1ms

            numpy_ego_gt_traj = npz_data['ego_gt_traj']
            tensor_ego_gt_traj = torch.from_numpy(numpy_ego_gt_traj).to(self.device)  # 3ms
            tensor_all_ego_gt_traj[npz_i, :npz_timestep] = tensor_ego_gt_traj  # 1ms

            numpy_ego_gt_traj_hist = npz_data['ego_gt_traj_hist']
            tensor_ego_gt_traj_hist = torch.from_numpy(numpy_ego_gt_traj_hist).to(self.device)  # 2.5ms
            tensor_all_ego_gt_traj_hist[npz_i, :npz_timestep] = tensor_ego_gt_traj_hist  # 1ms

            numpy_ego_gt_traj_long = npz_data['ego_gt_traj_long']
            tensor_ego_gt_traj_long = torch.from_numpy(numpy_ego_gt_traj_long).to(self.device)  # 3ms
            tensor_all_ego_gt_traj_long[npz_i, :npz_timestep] = tensor_ego_gt_traj_long  # 1ms
        print("npz 2 tensor time per bag (ms): ", (time.time() - start_time) * 1000 / self.all_bag_num)
        return (tensor_all_ego_gt_traj, tensor_all_ego_gt_traj_hist, tensor_all_ego_gt_traj_long,
                tensor_all_vectornet_object_feature, tensor_all_vectornet_object_mask,
                tensor_all_vectornet_static_feature)

    def __init__(self, device, scene_num, loop_mode):
        self.device = device

        list_str_path = self.get_fileName(scene_num)

        # file name 2 npz
        list_npz_data = self.trans_fileName_to_npz(list_str_path)

        # npz 2 numpy
        (tensor_all_ego_gt_traj,  # [10, 254, 20, 2]
         self.tensor_all_ego_gt_traj_hist,  # [10, 254, 10, 2]
         tensor_all_ego_gt_traj_long,  # [10, 254, 60, 2]
         self.tensor_all_vectornet_object_feature,  # [10, 254, 100, 16, 11]
         self.tensor_all_vectornet_object_mask,  # [10, 254, 100, 16]
         tensor_all_vectornet_static_feature) = (self.trans_npz_to_tensor(list_npz_data))  # [10, 254, 80, 16, 6]

        # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(202,))
        self.observation_space = "Dict"
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

        self.loop_mode = loop_mode

    def sort_dis(self, tensor_batch_obs_other):
        norm = torch.linalg.vector_norm(tensor_batch_obs_other[:, :, 0:2], dim=2)
        sorted_indices = torch.argsort(norm, dim=1)  # , descending=True
        tensor_batch_obs_other = torch.gather(tensor_batch_obs_other, 1,
                                              sorted_indices.unsqueeze(-1).expand(-1, -1, 4))  # [B, 99, 2]
        return tensor_batch_obs_other

    def mask_other(self, tensor_batch_obs_other, tensor_batch_obs_other_mask):
        replace_index = tensor_batch_obs_other_mask.unsqueeze(-1).repeat_interleave(4, dim=-1) == 0
        tensor_batch_obs_other = torch.where(
            replace_index,
            torch.tensor(9999.0, device=self.device).expand_as(tensor_batch_obs_other),
            tensor_batch_obs_other)  # [B, 99, 4]
        return tensor_batch_obs_other

    def obs_other(self):
        tensor_batch_obs_other_pos = self.tensor_batch_oneTime_other_pos_start_relaSim  # [B, 99, 2]
        tensor_batch_obs_other_vel = self.tensor_batch_oneTime_other_vel  # [B, 99, 2]
        tensor_batch_obs_other_mask = self.tensor_batch_oneTime_other_mask  # [B, 99]
        tensor_batch_obs_other = torch.cat([tensor_batch_obs_other_pos, tensor_batch_obs_other_vel],
                                           dim=2)  # [B, 99, 4]

        tensor_batch_obs_other = self.mask_other(tensor_batch_obs_other, tensor_batch_obs_other_mask)

        tensor_batch_obs_other = tensor_batch_obs_other[:, torch.randperm(tensor_batch_obs_other.size(1)), :]
        tensor_batch_obs_other = self.sort_dis(tensor_batch_obs_other)
        tensor_batch_obs_other = tensor_batch_obs_other[:, :50, :]  # [B, 50, 4]
        return tensor_batch_obs_other

    def obs_ego(self):
        tensor_batch_oneTime_sim_velocityX = torch.cos(
            self.tensor_batch_oneTime_sim_yaw) * self.tensor_batch_oneTime_sim_speed
        tensor_batch_oneTime_sim_velocityY = torch.sin(
            self.tensor_batch_oneTime_sim_yaw) * self.tensor_batch_oneTime_sim_speed
        # self.tensor_batch_oneTime_sim_posXYStart_relaEgo  # [B, 2]
        tensor_batch_ego = torch.cat([
            self.tensor_batch_oneTime_sim_speed.unsqueeze(1),  # [B, 1]
            self.tensor_batch_oneTime_sim_yaw.unsqueeze(1),  # [B, 1]
            tensor_batch_oneTime_sim_velocityX.unsqueeze(1),  # [B, 1]
            tensor_batch_oneTime_sim_velocityY.unsqueeze(1),  # [B, 1]
            # self.tensor_batch_oneTime_sim_posXYStart_relaEgo,  # [B, 2]  # cheat!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ], dim=1)  # [B, 6]

        return tensor_batch_ego

    def obs_cheat(self):
        tensor_batch_cheat = self.tensor_batch_oneTime_sim_posXYStart_relaEgo  # [B, 2]
        return tensor_batch_cheat

    def observe_once(self):
        tensor_batch_obs_other = self.obs_other()
        tensor_batch_ego = self.obs_ego()
        tensor_batch_cheat = self.obs_cheat()

        dict_tensor_batch_obs = {
            "tensor_batch_obs_other": tensor_batch_obs_other,
            "tensor_batch_ego": tensor_batch_ego,
            "tensor_batch_cheat": tensor_batch_cheat,
        }
        return dict_tensor_batch_obs

    def reset(self, batch_num, data_mode):  # ="Train"
        self.batch_num = batch_num
        if data_mode == "Train":
            # allTest_bag_num
            allTrain_bag_num = int(self.all_bag_num * 0.9)
            scene_indexes = list(range(allTrain_bag_num))
        elif data_mode == "Test":
            allTrain_bag_num = int(self.all_bag_num * 0.9)
            # allTest_bag_num = self.all_bag_num - allTrain_bag_num
            scene_indexes = list(range(allTrain_bag_num, self.all_bag_num))
        else:
            raise
        random.shuffle(scene_indexes)
        if self.batch_num > len(scene_indexes):
            self.batch_num = len(scene_indexes)
            print("self.batch_num is too large, limited to: ", self.batch_num)
        self.selected_scene_indexes = scene_indexes[:self.batch_num]
        # print("self.selected_scene_indexes: ", self.selected_scene_indexes)
        self.tensor_batch_vectornet_object_feature = self.tensor_all_vectornet_object_feature[
            self.selected_scene_indexes]  # [B, 254, 100, 16, 11]
        self.tensor_batch_vectornet_object_mask = self.tensor_all_vectornet_object_mask[
            self.selected_scene_indexes]  # [B, 254, 100, 16]
        self.tensor_batch_ego_gt_traj_hist = self.tensor_all_ego_gt_traj_hist[
            self.selected_scene_indexes]  # [B, 254, 10, 2]

        self.tensor_batch_oneTime_ego_posXYStart_relaStart = torch.zeros(self.batch_num, 2, device=self.device)
        if self.loop_mode == "Closed":
            self.tensor_batch_oneTime_sim_posXYStart_relaStart = torch.rand(self.batch_num, 2,
                                                                            device=self.device) * 2  # [B, 2]
        elif self.loop_mode == "Open":
            self.tensor_batch_oneTime_sim_posXYStart_relaStart = torch.zeros(self.batch_num, 2,
                                                                             device=self.device)  # [B, 2]
        else:
            raise
        self.tensor_batch_oneTime_sim_posXYStart_relaEgo = self.tensor_batch_oneTime_sim_posXYStart_relaStart - self.tensor_batch_oneTime_ego_posXYStart_relaStart  # [B, 2]

        # tensor_batch_obs = torch.tensor(
        #     [[[self.selected_scene_indexes[x], y] for y in range(254)] for x in range(self.batch_num)],
        #     device=self.device,
        #     dtype=torch.float)  # [20, 254, 2]
        self.timestep = 0

        self.tensor_batch_oneTime_other_pos_start_relaEgo = self.tensor_batch_vectornet_object_feature[:, self.timestep,
                                                            1:, 0, 0:2]
        self.tensor_batch_oneTime_other_pos_start_relaSim = self.tensor_batch_oneTime_other_pos_start_relaEgo - self.tensor_batch_oneTime_sim_posXYStart_relaStart.unsqueeze(
            1).repeat_interleave(99, dim=1)
        self.tensor_batch_oneTime_other_vel = self.tensor_batch_vectornet_object_feature[:, self.timestep, 1:, 0, 4:6]
        self.tensor_batch_oneTime_other_mask = self.tensor_batch_vectornet_object_mask[:, self.timestep, 1:,
                                               0]  # [B, 254, 100, 16] -> [B, 99]

        self.tensor_batch_oneTime_ego_velocity = self.tensor_batch_vectornet_object_feature[:, self.timestep, 0, 0,
                                                 4:6]  # [B, 2]
        # self.tensor_batch_oneTime_ego_yaw_arctan2 = torch.arctan2(self.tensor_batch_oneTime_ego_velocity[:, 1], self.tensor_batch_oneTime_ego_velocity[:, 0])  # not accurate
        self.tensor_batch_oneTime_ego_speed = torch.norm(self.tensor_batch_oneTime_ego_velocity, dim=-1)  # [B]
        self.tensor_batch_oneTime_ego_yaw = self.tensor_batch_vectornet_object_feature[:, self.timestep, 0, 0,
                                            10]  # [B]

        self.tensor_batch_oneTime_sim_speed = self.tensor_batch_oneTime_ego_speed
        self.tensor_batch_oneTime_sim_yaw = self.tensor_batch_oneTime_ego_yaw

        dict_tensor_batch_obs = self.observe_once()

        return dict_tensor_batch_obs

    def step_main_ego_pos(self):
        tensor_batch_oneTime_ego_deltaPosXYStart = - self.tensor_batch_ego_gt_traj_hist[:, self.timestep,
                                                     1] / 2  # [B, 2]
        self.tensor_batch_oneTime_ego_posXYStart_relaStart = self.tensor_batch_oneTime_ego_posXYStart_relaStart + tensor_batch_oneTime_ego_deltaPosXYStart  # [B, 2]

    def step_main_other_pos(self):
        self.tensor_batch_oneTime_other_pos_start_relaEgo = self.tensor_batch_vectornet_object_feature[:, self.timestep,
                                                            1:, 0, 0:2]
        self.tensor_batch_oneTime_other_vel = self.tensor_batch_vectornet_object_feature[:, self.timestep, 1:, 0, 4:6]
        self.tensor_batch_oneTime_other_mask = self.tensor_batch_vectornet_object_mask[:, self.timestep, 1:, 0]
        tensor_cpu_oneTime_other_pos_start_relaEgo = self.tensor_batch_oneTime_other_pos_start_relaEgo.cpu()  # [B, 99, 2]
        self.tensor_cpu_oneTime_other_pos_start_relaStart = tensor_cpu_oneTime_other_pos_start_relaEgo + \
                                                            self.tensor_batch_oneTime_ego_posXYStart_relaStart.cpu().detach().unsqueeze(
                                                                1).repeat_interleave(99,
                                                                                     dim=1)

    def step_main_other_posHis(self):
        tensor_oneTime_other_pos_his_start_relaEgo = self.tensor_batch_vectornet_object_feature[:, self.timestep, 1:,
                                                     :10, 0:2]  # [B, 99, 10, 2]
        if False:
            tensor_cpu_oneTime_other_pos_his_start_relaEgo = tensor_oneTime_other_pos_his_start_relaEgo.cpu()  # [99, 10, 2]
            tensor_cpu_oneTime_other_pos_his_start_relaEgo = tensor_cpu_oneTime_other_pos_his_start_relaEgo.reshape(990,
                                                                                                                    2)  # [990, 2]
            self.tensorCpu_oneTime_other_pos_his_start_relaStart = tensor_cpu_oneTime_other_pos_his_start_relaEgo + \
                                                                   self.tensor_batch_oneTime_ego_posXYStart_relaStart.cpu().detach().unsqueeze(
                                                                       1).repeat_interleave(990, dim=1)
        else:
            tensor_oneTime_other_pos_his_start_relaStart = tensor_oneTime_other_pos_his_start_relaEgo + self.tensor_batch_oneTime_ego_posXYStart_relaStart.unsqueeze(
                1).repeat_interleave(99, dim=1).unsqueeze(2).repeat_interleave(10, dim=2)  # [99, 10, 2]
            tensor_oneTime_other_pos_his_start_relaStart = tensor_oneTime_other_pos_his_start_relaStart.reshape(-1, 990,
                                                                                                                2)
            self.tensorCpu_oneTime_other_pos_his_start_relaStart = tensor_oneTime_other_pos_his_start_relaStart.cpu().detach()

    def step_main_ego_posHis(self):
        tensor_oneTime_ego_pos_his_start = self.tensor_batch_ego_gt_traj_hist[:, self.timestep]
        tensor_cpu_oneTime_ego_pos_his_start_relaEgo = tensor_oneTime_ego_pos_his_start.cpu()  # [10, 2]
        self.tensor_cpu_oneTime_ego_pos_his_start_relaStart = tensor_cpu_oneTime_ego_pos_his_start_relaEgo + \
                                                              self.tensor_batch_oneTime_ego_posXYStart_relaStart.cpu().detach().unsqueeze(
                                                                  1).repeat_interleave(10,
                                                                                       dim=1)

    def step_main_simulation(self):
        # simulation
        # if self.loop_mode == "Closed":
        self.tensor_batch_oneTime_sim_posXYStart_relaStart = self.tensor_batch_oneTime_sim_posXYStart_relaStart.detach() + self.tensor_batch_oneTime_action_deltaPosXy
        self.tensor_batch_oneTime_sim_posXYStart_relaEgo = self.tensor_batch_oneTime_sim_posXYStart_relaStart - self.tensor_batch_oneTime_ego_posXYStart_relaStart  # [B, 2]
        # elif self.loop_mode == "Open":
        #     self.tensor_batch_oneTime_sim_posXYStart_relaEgo = torch.zeros(self.batch_num, 2)
        #     self.tensor_batch_oneTime_sim_posXYStart_relaStart = self.tensor_batch_oneTime_sim_posXYStart_relaEgo + self.tensor_batch_oneTime_ego_posXYStart_relaStart
        # else:
        #     raise
        # self.tensor_batch_oneTime_other_pos_start_relaEgo  # [B, 99, 2]
        self.tensor_batch_oneTime_other_pos_start_relaSim = self.tensor_batch_oneTime_other_pos_start_relaEgo - self.tensor_batch_oneTime_sim_posXYStart_relaEgo.unsqueeze(
            1).repeat_interleave(99, dim=1)  # [B, 99, 2]

        self.tensor_batch_oneTime_sim_speed = torch.norm(self.tensor_batch_oneTime_action_deltaPosXy, dim=-1) / 0.1
        self.tensor_batch_oneTime_sim_yaw = torch.arctan2(self.tensor_batch_oneTime_action_deltaPosXy[:, 1],
                                                          self.tensor_batch_oneTime_action_deltaPosXy[:, 0])

    def step_main(self):
        self.timestep += 1
        self.step_main_ego_pos()  # calc tensor_batch_oneTime_ego_posXYStart_relaStart 自车位置
        self.step_main_other_pos()  # calc tensor_cpu_oneTime_other_pos_start_relaStart 周车位置
        self.step_main_ego_posHis()  # calc tensor_cpu_oneTime_ego_pos_his_start_relaStart 自车历史轨迹
        self.step_main_other_posHis()  # calc tensorCpu_oneTime_other_pos_his_start_relaStart 周车历史轨迹
        self.step_main_simulation()

    def calc_dis(self):
        tensor_batch_oneTime_other_dis_start_relaEgo = torch.norm(self.tensor_batch_oneTime_other_pos_start_relaEgo,
                                                                  dim=-1)  # [B, 99]
        tensor_batch_oneTime_other_dis_start_relaSim = torch.norm(self.tensor_batch_oneTime_other_pos_start_relaSim,
                                                                  dim=-1)  # [B, 99]

        temp_mask = torch.logical_and(self.tensor_batch_oneTime_other_pos_start_relaEgo[:, :, 0] != 0,
                                      self.tensor_batch_oneTime_other_pos_start_relaEgo[:, :, 1] != 0)

        tensor_batch_oneTime_other_dis_start_relaEgo = torch.where(temp_mask,
                                                                   tensor_batch_oneTime_other_dis_start_relaEgo,
                                                                   torch.tensor(999))  # [20, 99]
        tensor_batch_oneTime_other_dis_start_relaSim = torch.where(temp_mask,
                                                                   tensor_batch_oneTime_other_dis_start_relaSim,
                                                                   torch.tensor(999))  # [20, 99]

        self.tensor_batch_oneTime_dis_start_relaEgo, _ = torch.min(tensor_batch_oneTime_other_dis_start_relaEgo,
                                                                   dim=-1)  # [B]
        self.tensor_batch_oneTime_dis_start_relaSim, _ = torch.min(tensor_batch_oneTime_other_dis_start_relaSim,
                                                                   dim=-1)  # [B]

        return self.tensor_batch_oneTime_dis_start_relaEgo, self.tensor_batch_oneTime_dis_start_relaSim

    def calc_reward(self):
        # calc self.reward
        # calc dis with action
        self.calc_dis()
        dis_gt = torch.norm(self.tensor_batch_oneTime_sim_posXYStart_relaEgo, dim=-1)
        reward_gt = - dis_gt  # [B, 2] -inf~0
        reward_gt = torch.max(reward_gt, torch.ones_like(reward_gt) * -100)  # [B, 2] -100~0
        K_gt = 0.5  # bigger harder
        reward_gt_norm = 1 / (torch.exp(-K_gt * reward_gt))  # 0～1
        reward_safe = self.tensor_batch_oneTime_dis_start_relaSim  # [B, 2] 0~+inf
        K_safe = 0.1
        reward_safe_norm = 1 - torch.exp(-K_safe * reward_safe)
        self.reward = reward_gt_norm * W_gt + reward_safe_norm * W_safe
        loss = torch.square(dis_gt) * W_gt
        reward_info = {"reward_gt_norm": reward_gt_norm.detach(),
                       "reward_safe_norm": reward_safe_norm.detach(),
                       "loss": loss,
                       "dis_gt": dis_gt,
                       }
        return reward_info

    def step(self, tensor_batch_oneTime_action_deltaPosXy):
        # main step
        self.tensor_batch_oneTime_action_deltaPosXy = tensor_batch_oneTime_action_deltaPosXy
        self.step_main()

        reward_info = self.calc_reward()

        # calc done
        if self.timestep >= 253 - 1:  # 253 - 1
            done = True
        else:
            done = False

        # calc obs
        dict_tensor_batch_obs = self.observe_once()

        # calc info
        info = {}
        info.update(reward_info)

        if self.loop_mode == "Open":
            self.tensor_batch_oneTime_sim_posXYStart_relaEgo = torch.zeros(self.batch_num, 2, device=self.device)
            self.tensor_batch_oneTime_sim_posXYStart_relaStart = self.tensor_batch_oneTime_sim_posXYStart_relaEgo + self.tensor_batch_oneTime_ego_posXYStart_relaStart

        return self.reward, done, dict_tensor_batch_obs, info

    def render(self):
        plt.cla()

        plt.scatter(self.tensorCpu_oneTime_other_pos_his_start_relaStart[0, :, 0],
                    self.tensorCpu_oneTime_other_pos_his_start_relaStart[0, :, 1],
                    alpha=0.1, c="orange")  # 周车历史轨迹 Orange=Other

        plt.scatter(self.tensor_cpu_oneTime_ego_pos_his_start_relaStart[0, :, 0],
                    self.tensor_cpu_oneTime_ego_pos_his_start_relaStart[0, :, 1],
                    alpha=0.1, c="green")  # 自车gt历史轨迹 Green=eGo

        plt.scatter(self.tensor_cpu_oneTime_other_pos_start_relaStart[0, :, 0],
                    self.tensor_cpu_oneTime_other_pos_start_relaStart[0, :, 1],
                    alpha=0.5, c="orange")  # 周车当前位置 Orange=Other

        plt.scatter(self.tensor_batch_oneTime_ego_posXYStart_relaStart[0, 0].detach(),
                    self.tensor_batch_oneTime_ego_posXYStart_relaStart[0, 1].detach(),
                    alpha=0.5, c="green")  # 自车gt当前位置 Green=eGo

        plt.scatter(self.tensor_batch_oneTime_sim_posXYStart_relaStart[0, 0].detach(),
                    self.tensor_batch_oneTime_sim_posXYStart_relaStart[0, 1].detach(),
                    alpha=0.75, c="salmon", marker="*")  # 自车sim当前位置 Salmon=simulation

        plt.scatter(0, 0, alpha=0.5, c="skyblue")  # 自车起点位置 Skyblue=Start

        plt.xlim(-200, 200)
        plt.ylim(-200, 200)
        # plt.axis("equal")
        plt.pause(0.1)
