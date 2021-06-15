import os
from datetime import datetime as dt
from typing import List

import torch
import torch.nn.functional as f

from models.maddpg.ddpg import Agent
from models.maddpg.replay_buffer import ReplayBuffer
from utils.utils import soft_update


class MADDPG:

    def __init__(self, hp):

        self.hp = hp
        self.maddpg_agent = [Agent(self.hp, agent_id=0), Agent(self.hp, agent_id=1)]
        self.gamma = hp['gamma']
        self.tau = hp['tau']
        self.batch_size = hp['batch_size']
        self.seed = hp['random_seed']
        self.device = torch.device(hp['device'])
        self.update_every = hp['update_every']
        self.target_update_every = hp['target_update_every']
        self.learning_steps = hp['learning_steps']
        self.gradient_clipping_critic = hp['gradient_clipping_critic']
        self.gradient_clipping_actor = hp['gradient_clipping_actor']
        self.memory = ReplayBuffer(hp['replay_mem_size'], self.batch_size, self.seed,
                                   self.device)
        self.dir = f"{hp['results_path']}/{hp['env']}-{hp['model']}-{dt.now():%Y-%m-%d_%H:%M:%S}/"
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.checkpoint = f"{hp['results_path']}/{hp['env']}-{hp['model']}-{dt.now():%Y-%m-%d_%H:%M:%S}/checkpoints/"
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)
        self.l_steps = 0
        self.t_steps = 0

    def act(self, state, add_noise=True, damping_noise=1.0):
        actions = [agent.act(obs, add_noise, damping_noise) for agent, obs in zip(self.maddpg_agent, state)]
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)

        if len(self.memory) > self.batch_size:
            if self.l_steps % self.update_every == 0:
                for _ in range(self.learning_steps):
                    for a_i, a in enumerate(self.maddpg_agent):
                        experiences = self.memory.sample()
                        self.learn(experiences, a_i)
                    self.l_steps += 1
            if self.l_steps % self.target_update_every == 0:
                self.update_all_targets()

    def update_all_targets(self):
        for a in self.maddpg_agent:
            soft_update(a.critic_target, a.critic_local, self.tau)
            soft_update(a.actor_target, a.actor_local, self.tau)
        self.t_steps += 1

    def learn(self, samples, agent_i):
        """update the critics and actors of all the agents """
        states, actions, rewards, next_states, dones = samples
        states = torch.stack([torch.stack(s) for s in zip(*states)], dim=0)
        actions = torch.stack([torch.stack(s) for s in zip(*actions)], dim=0)
        rewards = torch.stack([torch.stack(s) for s in zip(*rewards)], dim=0)
        next_states = torch.stack([torch.stack(s) for s in zip(*next_states)], dim=0)
        dones = torch.stack([torch.stack(s) for s in zip(*dones)], dim=0)

        current_agent = self.maddpg_agent[agent_i]
        current_agent.critic_optimizer.zero_grad()
        all_target_actions = [policy(next_s) for policy, next_s in zip(self.get_target_actors(),
                                                                       next_states)]
        target_critic_input = torch.cat((*next_states, *all_target_actions), dim=1)

        target_value = (rewards[agent_i].view(-1, 1) + self.gamma *
                        current_agent.critic_target(target_critic_input) *
                        (1 - dones[agent_i].view(-1, 1)))

        local_critic_input = torch.cat((*states, *actions), dim=1)
        actual_value = current_agent.critic_local(local_critic_input)
        critic_loss = f.mse_loss(actual_value, target_value.detach())
        critic_loss.backward()
        if self.gradient_clipping_critic:
            torch.nn.utils.clip_grad_norm_(current_agent.critic_local.parameters(), 1)
        current_agent.critic_optimizer.step()

        current_agent.actor_optimizer.zero_grad()

        current_actor_output = current_agent.actor_local(states[agent_i])
        curr_pol_vf_in = current_actor_output
        all_policy_actions = []
        for i, policy, s in zip(range(2), self.get_actors(), states):
            if i == agent_i:
                all_policy_actions.append(curr_pol_vf_in)
            else:
                all_policy_actions.append(policy(s))
        local_critic_input = torch.cat((*states, *all_policy_actions), dim=1)
        pol_loss = -current_agent.critic_local(local_critic_input).mean()
        pol_loss += (current_actor_output ** 2).mean() * 1e-3
        pol_loss.backward()
        if self.gradient_clipping_actor:
            torch.nn.utils.clip_grad_norm_(current_agent.actor_local.parameters(), 1)
        current_agent.actor_optimizer.step()

    def get_actors(self):
        actors = [ddpg_agent.actor_local for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        target_actors = [ddpg_agent.actor_target for ddpg_agent in self.maddpg_agent]
        return target_actors

    def reset(self):
        [agent.reset() for agent in self.maddpg_agent]

    def load_weights(self, pth_path: List[str]):
        for a_i, a in enumerate(self.maddpg_agent):
            a.load_weights(pth_path[a_i])

    def save_weights(self, i_episode):
        for a_i, a in enumerate(self.maddpg_agent):
            a.save_weights(i_episode)
