import tqdm
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal

from utils_isaac_drive_env import IsaacDriveEnv

DEVICE = torch.device("cpu")  # cuda:0 cpu
NUM_EPOCH = 100
BATCH_NUM = 1


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
    for time_i in range(tensor_epoch_input.size(1)):
        if time_i == 0:
            tensor_time_output = tensor_epoch_input[:, 251 - time_i]
        else:
            tensor_time_output = tensor_epoch_input[:, 251 - time_i] + list_tensor_epoch_output[-1] * discount
        list_tensor_epoch_output.append(tensor_time_output)
    list_tensor_epoch_output = list_tensor_epoch_output[::-1]
    tensor_epoch_output = torch.stack(list_tensor_epoch_output, dim=1)
    return tensor_epoch_output


def generate_batch_actor(log_std, mu_net, tensor_batch_obs):
    tensor_batch_mu = mu_net(tensor_batch_obs)  # [B, 2]
    std = torch.exp(log_std)
    batch_pi = Normal(tensor_batch_mu, std)
    tensor_batch_action_xy = batch_pi.sample()
    tensor_batch_logp_a = batch_pi.log_prob(tensor_batch_action_xy).sum(axis=-1)
    return batch_pi, tensor_batch_action_xy, tensor_batch_logp_a


def main():
    env = IsaacDriveEnv(device=DEVICE)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    mu_net = mlp([obs_dim] + [64, 64] + [act_dim], activation=nn.Tanh)
    v_net = mlp([obs_dim] + [64, 64] + [1], activation=nn.Tanh)
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
            batch_pi, tensor_batch_action_xy, tensor_batch_logp_a = generate_batch_actor(log_std, mu_net,
                                                                                         tensor_batch_obs)
            tensor_batch_value = v_net(tensor_batch_obs)[0]
            tensor_batch_reward, bool_done, tensor_batch_obs_next = env.step(tensor_batch_action_xy)

            list_tensor_batch_obs.append(tensor_batch_obs)
            list_tensor_batch_action_xy.append(tensor_batch_action_xy)
            list_tensor_batch_reward.append(tensor_batch_reward)
            list_tensor_batch_value.append(tensor_batch_value)
            list_tensor_batch_logp_a.append(tensor_batch_logp_a)

            tensor_batch_obs = tensor_batch_obs_next

            if bool_done:
                tensor_batch_value_final = v_net(tensor_batch_obs)[0]
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

                tensor_epoch_obs = torch.stack(list_tensor_batch_obs, dim=1)
                tensor_epoch_action_xy = torch.stack(list_tensor_batch_action_xy, dim=1)
                tensor_epoch_logp_a = torch.stack(list_tensor_batch_logp_a, dim=1)

                train_pi_iters = 80
                for i in range(train_pi_iters):
                    pi_optimizer.zero_grad()
                    epoch_pi, tensor_epoch_action_xy_new, tensor_epoch_logp_a_new = generate_batch_actor(log_std,
                                                                                                         mu_net,
                                                                                                         tensor_epoch_obs)

                break

    print("Finished...")


if __name__ == '__main__':
    main()
