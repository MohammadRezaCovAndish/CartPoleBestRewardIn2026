import numpy as np
import random
from collections import deque
from typing import Tuple

class ReplayBuffer:
    """Experience Replay Buffer for DQN"""
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum size of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences from buffer"""
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays for efficiency
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training"""
        return len(self.buffer) >= batch_size