import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
from collections import namedtuple, deque

import numpy as np

import random

from models.maddpg.networks import Actor, Critic
from models.maddpg.replay_buffer import ReplayBuffer
from utils.noises import OUNoise


class Agent:

    def __init__(self, hp, agent_id=0):

        self.hp = hp
        self.agent_id = agent_id
        self.action_min = hp['action_min']
        self.action_max = hp['action_max']
        self.seed = hp['random_seed']
        self.device = torch.device(hp['device'])

        # Actor networks local and target
        self.actor_local = Actor(hp['state_size'], hp['action_size'], self.seed, hp['actor_layers']).to(self.device)
        self.actor_target = Actor(hp['state_size'], hp['action_size'], self.seed, hp['actor_layers']).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hp['actor_lr'])

        self.critic_local = Critic(hp['state_size'] * 2 + hp['action_size'] * 2, hp['action_size'], self.seed,
                                   hp['critic_layers']).to(self.device)
        self.critic_target = Critic(hp['state_size'] * 2 + hp['action_size'] * 2, hp['action_size'], self.seed,
                                    hp['critic_layers']).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hp['critic_lr'])

        self.noise = OUNoise((hp['action_size']), self.seed)

        # Create directory for experiment
        self.dir = f"{hp['results_path']}/{hp['env']}-{hp['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}/"
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.checkpoint = f"{hp['results_path']}/{hp['env']}-{hp['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}/checkpoints/"
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)

    def act(self, state, add_noise=True, damping_noise=1.0):
        state = torch.from_numpy(state).float().to(self.device)  # convert state from numpy array to a tensor
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * damping_noise
            action = np.clip(action, self.action_min, self.action_max)
        return action

    def reset(self):
        self.noise.reset()

    def load_weights(self, pth_path):
        self.actor_local.load_state_dict(torch.load(pth_path))
        self.actor_target.load_state_dict(torch.load(pth_path))

    def save_weights(self, i_episode):
        torch.save(self.actor_local.state_dict(), self.checkpoint + f'checkpoint_{i_episode}_{self.agent_id}.pth')
