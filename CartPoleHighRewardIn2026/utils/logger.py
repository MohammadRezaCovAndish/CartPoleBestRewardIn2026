import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json

class Logger:
    """Logger for training metrics"""
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        """
        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard
        """
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [],
            'losses': []
        }
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir)
        
        # Create metrics file
        self.metrics_file = os.path.join(log_dir, 'metrics.json')
        
    def log_episode(self, episode: int, reward: float, length: int, epsilon: float, loss: float = None):
        """Log episode metrics"""
        
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(length)
        self.metrics['epsilon_values'].append(epsilon)
        if loss is not None:
            self.metrics['losses'].append(loss)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.writer.add_scalar('Reward/episode', reward, episode)
            self.writer.add_scalar('Length/episode', length, episode)
            self.writer.add_scalar('Epsilon/episode', epsilon, episode)
            if loss is not None:
                self.writer.add_scalar('Loss/episode', loss, episode)
        
        # Print progress
        print(f"Episode {episode}: Reward = {reward:.2f}, Length = {length}, Epsilon = {epsilon:.3f}")
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, step)
            
    def save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
    def plot_rewards(self, save_path: str = None, window: int = 10):
        """Plot rewards with moving average"""
        
        rewards = self.metrics['episode_rewards']
        
        # Calculate moving average
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        else:
            moving_avg = rewards
        
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, alpha=0.6, label='Episode Reward')
        plt.plot(range(window-1, len(rewards)), moving_avg, 
                label=f'Moving Average (window={window})', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(self.log_dir, 'training_progress.png'))
        plt.close()
        
    def close(self):
        """Close logger"""
        if self.use_tensorboard:
            self.writer.close()
        self.save_metrics()
        self.plot_rewards()