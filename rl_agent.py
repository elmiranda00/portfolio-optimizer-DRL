# Implementation of the core Reinforcement Learning agent
# TEMP: based on vanilla Advantage Actor-Critic (A2C)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class ActorNetwork(nn.Module):
    """Actor network for policy gradient - map current portfolio state to a probability distribution"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class CriticNetwork(nn.Module):
    """Estimates the value function - the expected future return given the current portfolio state"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPOAgent:
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        # Optimizers to update Actor/Critic params
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Memory buffer to store experience (tuples)
        self.memory = deque(maxlen=10000)
        
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from current policy"""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state_tensor) # Get a distribution of possible allocations
        action = torch.multinomial(action_probs, 1).item()     # Sample action from distribution
        action_one_hot = np.zeros(action_probs.shape[1]) 
        action_one_hot[action] = 1.0 # One hot encode the sampled action
        
        return action_one_hot
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in memory"""

        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size: int = 32):
        """Train the agent"""
        
        if len(self.memory) < batch_size:
            return
        
        # Sample batch of experiences (sampled actions) from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # Calculate advantages of sampled actions w.r.t. baseline
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        next_values[dones] = 0
        
        targets = rewards + 0.95 * next_values
        advantages = targets - values
        
        # Update critic - minimize MSE between predicted values and obverved returns i.e. targets
        critic_loss = F.mse_loss(values, targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor - actions leading to higher than expected returns get higher probability
        action_probs = self.actor(states)
        action_log_probs = torch.log(torch.sum(action_probs * actions, dim=1) + 1e-8)
        actor_loss = -torch.mean(action_log_probs * advantages.detach())
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()