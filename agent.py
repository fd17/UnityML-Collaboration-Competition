import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn as nn
from torch.optim import Adam

from networks import Actor, Critic
from utils import OUNoise

LR_ACTOR = 3e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay - set to 0 to prevent sparse reward from drowning
device = "cpu"

class DDPG_agent(nn.Module):
    def __init__(self, in_actor, in_critic, action_size, num_agents, random_seed):
        super(DDPG_agent,self).__init__()
        """init the agent"""

        self.action_size = action_size
        self.seed = random_seed
        
        # Fully connected actor network
        self.actor_local = Actor(in_actor, self.action_size, self.seed).to(device)
        self.actor_target = Actor(in_actor, self.action_size, self.seed).to(device)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        

        # Fully connected critic network 
        self.critic_local = Critic(in_critic, num_agents*self.action_size, self.seed).to(device)
        self.critic_target = Critic(in_critic, num_agents*self.action_size, self.seed).to(device)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Ornstein-Uhlenbeck noise process for exploration
        self.noise = OUNoise((action_size), random_seed)
        
        
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device) 
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() 
        
        return np.clip(action, -1, 1)
    
    def target_act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        action = self.actor_target(state)
        return action
        
    def reset(self):
        """ Resets noise """
        self.noise.reset()