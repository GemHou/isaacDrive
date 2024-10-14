import time
import tqdm
import wandb
import torch
import torch.optim as optim
# from matplotlib import pyplot as plt

from utils_agent import Agent, AgentAcceleration, AgentVehicleDynamic
from utils_isaac_drive_env import IsaacDriveEnv

torch.autograd.set_detect_anomaly(True)

DEVICE = torch.device("cuda:0")  # cuda:0 cpu
RENDER_FLAG = True
SCENE_NUM = 2
TRAIN_BATCH_NUM = 1
TEST_BATCH_NUM = 1
# lr00005_randObs_sort_resetRandom2_removeCheat_
RESUME_NAME = "2024101422_9700K_grad_s2b1"  # 5700U 5900X 2070S 9700K 2080T
NUM_EPOCH = 100
TRAIN_LOOP_MODE = "Closed"  # Closed Open


def epoch_train(agent, isaac_drive_env, optimizer):
    isaac_drive_env.loop_mode = TRAIN_LOOP_MODE
    tensor_batch_obs = isaac_drive_env.reset(batch_num=TRAIN_BATCH_NUM, data_mode="Train")
    optimizer.zero_grad()
    list_tensor_time_loss = []
    list_tensor_time_reward_gt = []
    list_tensor_time_reward_safe = []
    list_dis_gt = []
    while True:
        # generate action
        if True:  # agent
            tensor_batch_action_xy = agent(tensor_batch_obs)  # [B, 2]
        else:  # random
            tensor_batch_action_xy = torch.randn(TRAIN_BATCH_NUM, 2, device=DEVICE)  # [B, 2]

        reward, done, tensor_batch_obs, info = isaac_drive_env.step(tensor_batch_action_xy)
        tensor_time_loss = info["loss"]
        list_tensor_time_loss.append(tensor_time_loss)
        list_tensor_time_reward_gt.append(info["reward_gt_norm"])
        list_tensor_time_reward_safe.append(info["reward_safe_norm"])
        list_dis_gt.append(info["dis_gt"])
        if done:
            break
    tensor_epoch_loss = torch.stack(list_tensor_time_loss, dim=1)
    tensor_epoch_reward_gt = torch.stack(list_tensor_time_reward_gt, dim=1)
    tensor_epoch_reward_safe = torch.stack(list_tensor_time_reward_safe, dim=1)
    tensor_epoch_dis_gt = torch.stack(list_dis_gt, dim=1)
    loss_final = tensor_epoch_loss.mean(dim=-1).mean()
    loss_per_step = loss_final.item()
    reward_gt = tensor_epoch_reward_gt.mean()
    reward_safe = tensor_epoch_reward_safe.mean()
    dis_gt = tensor_epoch_dis_gt.mean()
    class_name = "train" + TRAIN_LOOP_MODE
    wandb.log({class_name + "/dis_gt": dis_gt})
    wandb.log({class_name + "/loss_per_step": loss_per_step})
    wandb.log({class_name + "/reward_per_step_gt": reward_gt})
    wandb.log({class_name + "/reward_per_step_safe": reward_safe})
    loss_final.backward()
    optimizer.step()


def epoch_test(agent, isaac_drive_env, loop_mode, data_mode):
    isaac_drive_env.loop_mode = loop_mode
    dict_tensor_batch_obs = isaac_drive_env.reset(batch_num=TEST_BATCH_NUM, data_mode=data_mode)
    list_tensor_time_loss = []
    list_tensor_time_reward_gt = []
    list_tensor_time_reward_safe = []
    list_dis_gt = []
    while True:
        tensor_batch_action_xy = agent(dict_tensor_batch_obs)  # [B, 2]
        reward, done, dict_tensor_batch_obs, info = isaac_drive_env.step(tensor_batch_action_xy)
        tensor_time_loss = info["loss"]
        list_tensor_time_loss.append(tensor_time_loss)
        list_tensor_time_reward_gt.append(info["reward_gt_norm"])
        list_tensor_time_reward_safe.append(info["reward_safe_norm"])
        list_dis_gt.append(info["dis_gt"])
        if done:
            break
    tensor_epoch_loss = torch.stack(list_tensor_time_loss, dim=1)
    tensor_epoch_reward_gt = torch.stack(list_tensor_time_reward_gt, dim=1)
    tensor_epoch_reward_safe = torch.stack(list_tensor_time_reward_safe, dim=1)
    tensor_epoch_dis_gt = torch.stack(list_dis_gt, dim=1)
    loss_final = tensor_epoch_loss.mean(dim=-1).mean()
    loss_per_step = loss_final.item()
    reward_gt = tensor_epoch_reward_gt.mean()
    reward_safe = tensor_epoch_reward_safe.mean()
    dis_gt = tensor_epoch_dis_gt.mean()
    if data_mode == "Test":
        class_name = "test" + loop_mode
    elif data_mode == "Train":
        class_name = "train" + loop_mode
    else:
        raise
    wandb.log({class_name + "/dis_gt": dis_gt})
    wandb.log({class_name + "/loss_per_step": loss_per_step})
    wandb.log({class_name + "/reward_per_step_gt": reward_gt})
    wandb.log({class_name + "/reward_per_step_safe": reward_safe})


def main():
    wandb.init(
        project="isaac_drive",
        resume=RESUME_NAME  # HjScenarioEnv
    )

    isaac_drive_env = IsaacDriveEnv(device=DEVICE, scene_num=SCENE_NUM, loop_mode=TRAIN_LOOP_MODE)
    # obs_dim = isaac_drive_env.observation_space.shape[0]
    # agent = Agent(obs_dim=obs_dim)
    agent = AgentAcceleration()  # obs_dim=obs_dim
    # agent = AgentVehicleDynamic(obs_dim=obs_dim)
    if False:
        state_dict = torch.load("./data/interim/state_dict_grad.pt", map_location=DEVICE)
        agent.load_state_dict(state_dict)
    agent.to(DEVICE)
    lr = 0.0005
    num_epoch = NUM_EPOCH
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    start_time = time.time()

    for _ in tqdm.tqdm(range(num_epoch)):
        epoch_train(agent, isaac_drive_env, optimizer)
        epoch_test(agent, isaac_drive_env, loop_mode="Open", data_mode="Test")
        epoch_test(agent, isaac_drive_env, loop_mode="Closed", data_mode="Test")
        if TRAIN_LOOP_MODE == "Closed":
            epoch_test(agent, isaac_drive_env, loop_mode="Open", data_mode="Train")
        elif TRAIN_LOOP_MODE == "Open":
            epoch_test(agent, isaac_drive_env, loop_mode="Closed", data_mode="Train")
        else:
            raise
        torch.save(agent.state_dict(), "./data/interim/state_dict_grad.pt")

    print("update network time: ", time.time() - start_time)  # 15 second
    if not RENDER_FLAG:
        total_frame = 100 * TRAIN_BATCH_NUM * 253
        print("throughput: ", total_frame / (time.time() - start_time))

    torch.save(agent.state_dict(), "./data/interim/state_dict_grad.pt")

    print("Finished...")


if __name__ == '__main__':
    main()
