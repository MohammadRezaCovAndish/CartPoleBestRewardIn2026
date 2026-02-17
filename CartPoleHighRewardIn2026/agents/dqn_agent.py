import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
import random

from agents.networks import DQN, DuelingDQN
from agents.replay_buffer import ReplayBuffer

class DQNAgent:
    """DQN Agent for CartPole"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
            device: Device to run computations on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Hyperparameters
        self.gamma = config['training']['gamma']
        self.epsilon = config['training']['epsilon_start']
        self.epsilon_min = config['training']['epsilon_end']
        self.epsilon_decay = config['training']['epsilon_decay']
        self.learning_rate = config['training']['learning_rate']
        self.batch_size = config['training']['batch_size']
        self.target_update_freq = config['training']['target_update']
        self.warmup_steps = config['training']['warmup_steps']
        
        # Networks
        if config['model']['dueling']:
            self.q_network = DuelingDQN(
                state_dim, action_dim, config['model']['hidden_layers']
            ).to(device)
            self.target_network = DuelingDQN(
                state_dim, action_dim, config['model']['hidden_layers']
            ).to(device)
        else:
            self.q_network = DQN(
                state_dim, action_dim, config['model']['hidden_layers']
            ).to(device)
            self.target_network = DQN(
                state_dim, action_dim, config['model']['hidden_layers']
            ).to(device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.memory = ReplayBuffer(config['training']['memory_size'])
        
        # Training steps counter
        self.steps = 0
        self.episodes = 0
        
    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy policy"""
        
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def step(self, state, action, reward, next_state, done):
        """Store experience and train if ready"""
        
        # Store in replay buffer
        self.memory.push(state, action, reward, next_state, done)
        
        # Train if enough samples
        if self.memory.is_ready(self.batch_size) and self.steps > self.warmup_steps:
            self._train()
        
        self.steps += 1
        
    def _train(self):
        """Perform one training step"""
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
    def update_target_network(self):
        """Update target network with current Q network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'episode': self.episodes,
            'steps': self.steps,
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            }
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episodes = checkpoint['episode']
        self.steps = checkpoint['steps']
        print(f"Model loaded from {path}")