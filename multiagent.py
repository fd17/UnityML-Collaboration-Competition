
import numpy as np
import torch
import torch.nn.functional as F
from agent import DDPG_agent
from utils import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 3e-2              # for soft update of target parameters

class MADDPG():
    def __init__(self, state_size, action_size, num_agents, random_seed=0):
        in_critic = num_agents*state_size
        self.agents = [DDPG_agent(state_size, in_critic, action_size, num_agents, random_seed) for i in range(num_agents)]
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.num_agents = num_agents
        
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = [agent.act(state, add_noise) for agent, state in zip(self.agents, states)]
        return actions
    
    def target_act(self, states):
        """Returns actions for given state as per current policy."""
        actions = [agent.target_act(state) for agent, state in zip(self.agents, states)]
        return actions
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        #for i in range(state.shape[0]):
        state = np.asanyarray(state)
        action = np.asanyarray(action)
        reward = np.asanyarray(reward)
        next_state = np.asanyarray(next_state)
        done = np.asanyarray(done)
        self.memory.add(state.reshape((1, self.num_agents, -1)), action.reshape((1, self.num_agents, -1)), \
                        reward.reshape((1, self.num_agents, -1)), next_state.reshape((1,self.num_agents, -1)), \
                        done.reshape((1, self.num_agents, -1)))
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            for i_agent in range(self.num_agents):
                experiences = self.memory.sample()
                self.learn(experiences, i_agent, GAMMA)
    
    def reset(self):
        [agent.reset() for agent in self.agents]
        
    def learn(self, experiences, i_agent, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences

        agent = self.agents[i_agent]
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        next_states = next_states.view(1, BATCH_SIZE, self.num_agents, -1)
        actions_next = torch.cat(self.target_act(next_states), dim=1)
        next_states = next_states.view(BATCH_SIZE,-1)
        actions_next = actions_next.view(BATCH_SIZE,-1)
        
        Q_targets_next = agent.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards[:,i_agent] + (gamma * Q_targets_next * (1 - dones[:,i_agent]))
        # Compute critic loss
        Q_expected = agent.critic_local(states.view(BATCH_SIZE,-1), actions.view(BATCH_SIZE,-1))
        # mean squared error loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        # zero_grad because we do not want to accumulate 
        # gradients from other batches, so needs to be cleared
        agent.critic_optimizer.zero_grad()
        # compute derivatives for all variables that
        # requires_grad-True
        critic_loss.backward()
        # update those variables that requires_grad-True
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # take the current states and predict actions
        #states = states.view(1, BATCH_SIZE, self.num_agents, -1)
        actions_pred = agent.actor_local(states)
        #print (actions_pred.shape)
        #actions_pred = torch.cat(actions_pred, dim=1)
        # -1 * (maximize) Q value for the current prediction
        actor_loss = -agent.critic_local(states.view(BATCH_SIZE,-1), actions_pred.view(BATCH_SIZE,-1)).mean()
        # Minimize the loss
        # zero_grad because we do not want to accumulate 
        # gradients from other batches, so needs to be cleared
        agent.actor_optimizer.zero_grad()
        # compute derivatives for all variables that
        # requires_grad-True
        actor_loss.backward()
        # update those variables that requires_grad-True
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(agent.critic_local, agent.critic_target, TAU)
        self.soft_update(agent.actor_local, agent.actor_target, TAU) 
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)