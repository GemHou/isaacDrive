import time
import torch
import random
import numpy as np
from matplotlib import pyplot as plt


class IsaacDriveEnv:
    def trans_fileName_to_npz(self):
        start_time = time.time()
        list_str_path = [
            "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
            # "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
            # "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
            # "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
            # "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
            # "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
            # "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
            # "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
            # "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
            # "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        ]
        # list_str_path = list_str_path[:BAG_NUM]
        self.all_bag_num = len(list_str_path)
        list_npz_data = []
        for str_path in list_str_path:
            npz_data = np.load(str_path, allow_pickle=False)
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

    def __init__(self, device):
        self.device = device

        # file name 2 npz
        list_npz_data = self.trans_fileName_to_npz()

        # npz 2 numpy
        (tensor_all_ego_gt_traj,  # [10, 254, 20, 2]
         self.tensor_all_ego_gt_traj_hist,  # [10, 254, 10, 2]
         tensor_all_ego_gt_traj_long,  # [10, 254, 60, 2]
         self.tensor_all_vectornet_object_feature,  # [10, 254, 100, 16, 11]
         tensor_all_vectornet_object_mask,  # [10, 254, 100, 16]
         tensor_all_vectornet_static_feature) = (self.trans_npz_to_tensor(list_npz_data))  # [10, 254, 80, 16, 6]

    def reset(self, batch_num):
        self.batch_num = batch_num
        scene_indexes = list(range(self.all_bag_num))
        random.shuffle(scene_indexes)
        self.selected_scene_indexes = scene_indexes[:self.batch_num]

        # tensor_batch_obs = torch.tensor(
        #     [[[self.selected_scene_indexes[x], y] for y in range(254)] for x in range(self.batch_num)],
        #     device=self.device,
        #     dtype=torch.float)  # [20, 254, 2]
        self.timestep = 0
        tensor_batch_obs = torch.tensor([[self.selected_scene_indexes[x], self.timestep] for x in range(self.batch_num)],
                                        device=self.device, dtype=torch.float)  # [20, 2]
        self.tensor_batch_vectornet_object_feature = self.tensor_all_vectornet_object_feature[
            self.selected_scene_indexes]  # [B, 254, 100, 16, 11]
        self.tensor_batch_ego_gt_traj_hist = self.tensor_all_ego_gt_traj_hist[self.selected_scene_indexes]  # [B, 254, 10, 2]

        self.delta_xy = torch.zeros(self.batch_num, 2, device=self.device)

        return tensor_batch_obs

    def calc_dis(self):
        start_time = time.time()
        tensor_batch_ego_pos_start = self.tensor_batch_vectornet_object_feature[:, :, 0, 0, 0:2]  # [20, 254, 2]
        assert torch.all(tensor_batch_ego_pos_start == 0)
        tensor_batch_other_pos_start = self.tensor_batch_vectornet_object_feature[:, :, 1:, 0, 0:2]  # [20, 254, 99, 2]
        tensor_batch_other_dis_start = torch.norm(tensor_batch_other_pos_start, dim=-1)  # [20, 254, 99]
        tensor_batch_other_dis_start = torch.where(tensor_batch_other_dis_start != 0, tensor_batch_other_dis_start,
                                                   torch.tensor(999))  # [20, 254, 99]
        tensor_batch_dis_start, _ = torch.min(tensor_batch_other_dis_start, dim=-1)  # [20, 254]

        print("calc dis time per bag (ms)", (time.time() - start_time) * 1000 / self.batch_num)

        return tensor_batch_dis_start

    def calc_dis_withAction(self):
        tensor_batch_oneTime_ego_pos_start = self.tensor_batch_oneTime_action_xy  # [20, 2]
        tensor_batch_oneTime_ego_repeat_pos_start = tensor_batch_oneTime_ego_pos_start.unsqueeze(1)  # [20, 1, 2]
        tensor_batch_oneTime_ego_repeat_pos_start = tensor_batch_oneTime_ego_repeat_pos_start.repeat_interleave(99,
                                                                                                                1)  # [20, 99, 2]
        tensor_batch_oneTime_other_pos_start = self.tensor_batch_vectornet_object_feature[:, self.timestep, 1:, 0,
                                               0:2]  # [20, 99, 2]
        temp_mask = torch.logical_and(tensor_batch_oneTime_other_pos_start[:, :, 0] != 0,
                                      tensor_batch_oneTime_other_pos_start[:, :, 1] != 0)

        tensor_batch_oneTime_other_dis_start_withAction = torch.norm(
            tensor_batch_oneTime_other_pos_start - tensor_batch_oneTime_ego_repeat_pos_start,
            dim=-1)  # [20, 254, 99]
        tensor_batch_oneTime_other_dis_start_withAction = torch.where(temp_mask,
                                                                      tensor_batch_oneTime_other_dis_start_withAction,
                                                                      torch.tensor(999))  # [20, 254, 99]
        tensor_batch_oneTime_dis_start_withAction, _ = torch.min(tensor_batch_oneTime_other_dis_start_withAction,
                                                                 dim=-1)  # [20, 254]

        tensor_batch_oneTime_other_dis_start_woAction = torch.norm(tensor_batch_oneTime_other_pos_start,
                                                                   dim=-1)  # [20, 254, 99]
        tensor_batch_oneTime_other_dis_start_woAction = torch.where(temp_mask,
                                                                    tensor_batch_oneTime_other_dis_start_woAction,
                                                                    torch.tensor(999))  # [20, 254, 99]
        tensor_batch_oneTime_dis_start_woAction, _ = torch.min(tensor_batch_oneTime_other_dis_start_woAction,
                                                               dim=-1)  # [20, 254]

        return tensor_batch_oneTime_dis_start_withAction, tensor_batch_oneTime_dis_start_woAction

    def step(self,
             tensor_batch_oneTime_action_xy
             ):
        self.timestep += 1

        self.delta_xy += tensor_batch_oneTime_action_xy

        self.tensor_batch_oneTime_action_xy = tensor_batch_oneTime_action_xy
        # calc dis with action
        tensor_batch_oneTime_dis_start_withAction, tensor_batch_oneTime_dis_start_woAction = self.calc_dis_withAction()
        reward = tensor_batch_oneTime_dis_start_withAction - tensor_batch_oneTime_dis_start_woAction

        if self.timestep >= 254 - 1:
            done = True
        else:
            done = False

        tensor_batch_obs = torch.tensor([[self.selected_scene_indexes[x], self.timestep] for x in range(self.batch_num)],
                                        device=self.device, dtype=torch.float)  # [20, 2]

        return reward, done, tensor_batch_obs

    def render(self):
        plt.cla()

        tensor_oneTime_other_pos_his_start = self.tensor_batch_vectornet_object_feature[0, self.timestep, 1:, :10, 0:2]
        tensor_cpu_oneTime_other_pos_his_start = tensor_oneTime_other_pos_his_start.cpu()  # [99, 10, 2]
        tensor_cpu_oneTime_other_pos_his_start = tensor_cpu_oneTime_other_pos_his_start.reshape(990, 2)
        tensor_cpu_oneTime_other_pos_his_start -= self.delta_xy[0].cpu().detach().unsqueeze(0).repeat_interleave(990, dim=0)
        plt.scatter(tensor_cpu_oneTime_other_pos_his_start[:, 0], tensor_cpu_oneTime_other_pos_his_start[:, 1], alpha=0.1)

        tensor_oneTime_ego_pos_his_start = self.tensor_batch_ego_gt_traj_hist[0, self.timestep]
        tensor_cpu_oneTime_ego_pos_his_start = tensor_oneTime_ego_pos_his_start.cpu()  # [10, 2]
        tensor_cpu_oneTime_ego_pos_his_start -= self.delta_xy[0].cpu().detach().unsqueeze(0).repeat_interleave(10, dim=0)
        plt.scatter(tensor_cpu_oneTime_ego_pos_his_start[:, 0], tensor_cpu_oneTime_ego_pos_his_start[:, 1], alpha=0.1)

        tensor_oneTime_other_pos_start = self.tensor_batch_vectornet_object_feature[0, self.timestep, 1:, 0, 0:2]
        tensor_cpu_oneTime_other_pos_start = tensor_oneTime_other_pos_start.cpu()  # [99, 2]
        tensor_cpu_oneTime_other_pos_start -= self.delta_xy[0].cpu().detach().unsqueeze(0).repeat_interleave(99, dim=0)
        plt.scatter(tensor_cpu_oneTime_other_pos_start[:, 0], tensor_cpu_oneTime_other_pos_start[:, 1])

        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        numpy_oneTime_action_xy = self.tensor_batch_oneTime_action_xy[0].cpu().detach().numpy()
        plt.plot([0, numpy_oneTime_action_xy[0]], [0, numpy_oneTime_action_xy[1]], "r")
        plt.pause(0.1)
