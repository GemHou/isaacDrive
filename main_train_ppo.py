import tqdm
import torch
import wandb
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal

from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu
NUM_EPOCH = 1000
BATCH_NUM = 100
RESUME_NAME = "ppo_s100b100_20241014"


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def discount_cumsum(tensor_epoch_input, discount):
    """
    computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    list_tensor_epoch_output = []
    time_length = tensor_epoch_input.size(1)
    for time_i in range(time_length):
        if time_i == 0:
            tensor_time_output = tensor_epoch_input[:, time_length - 1 - time_i]
        else:
            tensor_time_output = tensor_epoch_input[:, time_length - 1 - time_i] + list_tensor_epoch_output[
                -1] * discount
        list_tensor_epoch_output.append(tensor_time_output)
    list_tensor_epoch_output = list_tensor_epoch_output[::-1]
    tensor_epoch_output = torch.stack(list_tensor_epoch_output, dim=1)
    return tensor_epoch_output


def generate_batch_actor(log_std, mu_net, tensor_batch_obs):
    tensor_batch_mu = mu_net(tensor_batch_obs) * 5  # [B, 2]
    std = torch.exp(log_std)
    batch_pi = Normal(tensor_batch_mu, std)
    tensor_batch_action_xy = batch_pi.sample()
    tensor_batch_logp_a = batch_pi.log_prob(tensor_batch_action_xy).sum(axis=-1)
    return tensor_batch_action_xy, tensor_batch_logp_a


def collect_experience_step(env, list_tensor_batch_action_xy, list_tensor_batch_logp_a, list_tensor_batch_obs,
                            list_tensor_batch_reward, list_tensor_batch_value, log_std, mu_net, tensor_batch_obs,
                            v_net):
    with torch.no_grad():
        tensor_batch_action_xy, tensor_batch_logp_a = generate_batch_actor(log_std, mu_net,
                                                                           tensor_batch_obs)
        tensor_batch_value = v_net(tensor_batch_obs)[:, 0]
        tensor_batch_reward, bool_done, tensor_batch_obs_next, info = env.step(tensor_batch_action_xy)
        list_tensor_batch_obs.append(tensor_batch_obs)
        list_tensor_batch_action_xy.append(tensor_batch_action_xy)
        list_tensor_batch_reward.append(tensor_batch_reward)
        list_tensor_batch_value.append(tensor_batch_value)
        list_tensor_batch_logp_a.append(tensor_batch_logp_a)
        tensor_batch_obs = tensor_batch_obs_next
    return bool_done, tensor_batch_obs


def finish_path(list_tensor_batch_action_xy, list_tensor_batch_logp_a, list_tensor_batch_obs, list_tensor_batch_reward,
                list_tensor_batch_value, tensor_batch_obs, v_net):
    with torch.no_grad():
        tensor_batch_value_final = v_net(tensor_batch_obs)[:, 0]
        list_tensor_batch_reward.append(tensor_batch_value_final)
        list_tensor_batch_value.append(tensor_batch_value_final)

        tensor_epoch_reward = torch.stack(list_tensor_batch_reward, dim=1)  # [B, T+1]
        tensor_epoch_value = torch.stack(list_tensor_batch_value, dim=1)  # [B, T+1]

        gamma = 0.99
        lam = 0.97
        tensor_epoch_deltas = tensor_epoch_reward[:, :-1] + gamma * tensor_epoch_value[:,
                                                                    1:] - tensor_epoch_value[:, :-1]
        tensor_epoch_adv = discount_cumsum(tensor_epoch_deltas, discount=gamma * lam)
        tensor_epoch_ret = discount_cumsum(tensor_epoch_reward, discount=gamma)
        tensor_epoch_retWoDiscount = torch.sum(tensor_epoch_reward, dim=-1)
        # print("tensor_epoch_retWoDiscountL: ", tensor_epoch_retWoDiscount)

        tensor_epoch_obs = torch.stack(list_tensor_batch_obs, dim=1)
        tensor_epoch_action_xy = torch.stack(list_tensor_batch_action_xy, dim=1)
        tensor_epoch_logp_a = torch.stack(list_tensor_batch_logp_a, dim=1)
    return tensor_epoch_action_xy, tensor_epoch_adv, tensor_epoch_logp_a, tensor_epoch_obs, tensor_epoch_ret, tensor_epoch_retWoDiscount


def update_p(log_std, mu_net, pi_optimizer, tensor_epoch_action_xy, tensor_epoch_adv, tensor_epoch_logp_a,
             tensor_epoch_obs):
    pi_optimizer.zero_grad()
    tensor_epoch_mu = mu_net(tensor_epoch_obs) * 5  # [B, 2]
    std = torch.exp(log_std)
    epoch_pi = Normal(tensor_epoch_mu, std)
    tensor_epoch_logp_a_new = epoch_pi.log_prob(tensor_epoch_action_xy).sum(axis=-1)
    tensor_epoch_ratio = torch.exp(tensor_epoch_logp_a_new - tensor_epoch_logp_a)
    clip_ratio = 0.2
    tensor_epoch_clip_adv = torch.clamp(tensor_epoch_ratio, 1 - clip_ratio, 1 + clip_ratio) * tensor_epoch_adv
    loss_pi = -(torch.min(tensor_epoch_ratio * tensor_epoch_adv, tensor_epoch_clip_adv)).mean()
    # print("loss_pi: ", loss_pi)
    loss_pi.backward()

    float_approx_kl = (tensor_epoch_logp_a - tensor_epoch_logp_a_new).mean().item()
    pi_optimizer.step()
    return float_approx_kl


def update_v(tensor_epoch_obs, tensor_epoch_ret, v_net, vf_optimizer):
    vf_optimizer.zero_grad()
    new_v = v_net(tensor_epoch_obs)[:, :, 0]
    loss_v = ((new_v - tensor_epoch_ret[:, :-1]) ** 2).mean()
    # print("loss_v: ", loss_v)
    loss_v.backward()
    vf_optimizer.step()


def main():
    wandb.init(
        project="isaac_drive",
        resume=RESUME_NAME  # HjScenarioEnv
    )

    env = IsaacDriveEnv(device=DEVICE)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    mu_net = mlp([obs_dim] + [64, 64, 64, 64] + [act_dim], activation=nn.Tanh, output_activation=nn.Tanh)
    v_net = mlp([obs_dim] + [64, 64, 64, 64] + [1], activation=nn.Tanh)
    log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
    log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    pi_lr = 3e-4
    vf_lr = 1e-3
    pi_optimizer = Adam([{'params': mu_net.parameters()},
                         {'params': log_std}
                         ],
                        lr=pi_lr)
    vf_optimizer = Adam(v_net.parameters(), lr=vf_lr)

    for epoch in tqdm.tqdm(range(NUM_EPOCH)):
        list_tensor_batch_obs = []
        list_tensor_batch_action_xy = []
        list_tensor_batch_reward = []
        list_tensor_batch_value = []
        list_tensor_batch_logp_a = []

        tensor_batch_obs = env.reset(batch_num=BATCH_NUM)
        while True:
            bool_done, tensor_batch_obs = collect_experience_step(env, list_tensor_batch_action_xy,
                                                                  list_tensor_batch_logp_a, list_tensor_batch_obs,
                                                                  list_tensor_batch_reward, list_tensor_batch_value,
                                                                  log_std, mu_net, tensor_batch_obs, v_net)

            if bool_done:
                tensor_epoch_action_xy, tensor_epoch_adv, tensor_epoch_logp_a, \
                    tensor_epoch_obs, tensor_epoch_ret, tensor_epoch_retWoDiscount = finish_path(
                    list_tensor_batch_action_xy, list_tensor_batch_logp_a, list_tensor_batch_obs,
                    list_tensor_batch_reward, list_tensor_batch_value, tensor_batch_obs, v_net)

                wandb.log({"return_epoch": tensor_epoch_retWoDiscount.mean().detach()})

                train_pi_iters = 80
                for i in range(train_pi_iters):
                    float_approx_kl = update_p(log_std, mu_net, pi_optimizer, tensor_epoch_action_xy, tensor_epoch_adv,
                                               tensor_epoch_logp_a, tensor_epoch_obs)
                    target_kl = 0.01
                    if float_approx_kl > 1.5 * target_kl:
                        print("early stop at step: ", i)
                        break

                train_v_iters = 80
                for i in range(train_v_iters):
                    update_v(tensor_epoch_obs, tensor_epoch_ret, v_net, vf_optimizer)

                break

    print("Finished...")


if __name__ == '__main__':
    main()
