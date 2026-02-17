import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    """Deep Q-Network for CartPole"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list = [128, 128]):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_layers: List of hidden layer sizes
        """
        super(DQN, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        # Build hidden layers without BatchNorm
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        features = self.feature_layer(x)
        return self.output_layer(features)


class DuelingDQN(nn.Module):
    """Dueling DQN Architecture for CartPole"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list = [128, 128]):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_layers: List of hidden layer sizes
        """
        super(DuelingDQN, self).__init__()
        
        # Shared feature layers without BatchNorm
        feature_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_layers:
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*feature_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass with dueling architecture"""
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        # Handle single sample case by adding batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values