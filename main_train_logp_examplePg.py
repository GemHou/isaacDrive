import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2,
          epochs=500, batch_size=5000, render=True):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    # assert isinstance(env.action_space, Discrete), \
    #     "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, Discrete):
        n_discrete_acts = env.action_space.n
        # make core of policy network
        logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_discrete_acts])

        # make function to compute action distribution
        def get_discrete_policy(obs):
            logits = logits_net(obs)
            return Categorical(logits=logits)

        get_policy = get_discrete_policy

        # make action selection function (outputs int actions, sampled from policy)
        def get_action(obs):
            return get_policy(obs).sample().item()

        # make optimizer
        optimizer = Adam(logits_net.parameters(), lr=lr)
    else:
        n_continuous_acts = env.action_space.shape[0]
        # make core of policy network
        mu_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_continuous_acts])
        log_std = -0.5 * np.ones(n_continuous_acts, dtype=np.float32)
        log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # make function to compute action distribution
        def get_continuous_policy(obs):
            mu = mu_net(obs)
            std = torch.exp(log_std)
            return Normal(mu, std)

        get_policy = get_continuous_policy

        # make action selection function (outputs int actions, sampled from policy)
        def get_action(obs):
            return get_policy(obs).sample().item()

        # make optimizer
        optimizer = Adam(mu_net.parameters(), lr=lr)

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs, _ = env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep
        ep_len = 0

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if render:  # (not finished_rendering_this_epoch) and
                env.render()
                # print("render")
                # plt.pause(0.00000001)

            # save obs
            obs_copy = copy.deepcopy(obs)
            batch_obs.append(obs_copy)

            # act in the environment
            tensor_obs = torch.as_tensor(obs, dtype=torch.float32)
            act = get_action(tensor_obs)
            # print("act: ", act)
            obs, rew, done, _, _ = env.step([act])

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)
            ep_len += 1

            if done or ep_len > 1000:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                (obs, _), done, ep_rews, ep_len = env.reset(), False, [], 0

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='Pendulum-v1')  # CartPole-v0
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
