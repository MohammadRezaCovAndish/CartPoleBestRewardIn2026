#!/usr/bin/env python3
"""
Evaluate trained model with detailed statistics
"""

import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
import torch

from agents.dqn_agent import DQNAgent

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate_agent(agent, env, num_episodes=100):
    """Evaluate agent with detailed statistics"""
    
    rewards = []
    success_rate = 0
    success_threshold = 475  # CartPole solved at 475+ reward
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.act(state, eval_mode=True)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
        
        if episode_reward >= success_threshold:
            success_rate += 1
    
    success_rate = (success_rate / num_episodes) * 100
    
    # Calculate statistics
    stats = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'success_rate': success_rate,
        'rewards': rewards
    }
    
    return stats

def plot_results(stats, save_path='evaluation_results.png'):
    """Plot evaluation results"""
    
    plt.figure(figsize=(10, 6))
    
    # Rewards histogram
    plt.hist(stats['rewards'], bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(stats['mean_reward'], color='r', linestyle='--', 
                label=f"Mean: {stats['mean_reward']:.2f}")
    plt.axvline(475, color='g', linestyle='--', label='Success Threshold (475)')
    
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Evaluation Results Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text with statistics
    text = f"Statistics:\n"
    text += f"Mean: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}\n"
    text += f"Max: {stats['max_reward']:.2f}\n"
    text += f"Min: {stats['min_reward']:.2f}\n"
    text += f"Success Rate: {stats['success_rate']:.1f}%"
    
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Evaluation plot saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DQN model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create environment
    env = gym.make(
        config['environment']['name'],
        max_episode_steps=config['environment']['max_episode_steps']
    )
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_dim, action_dim, config, device)
    
    # Load model
    agent.load(args.model)
    
    # Evaluate
    stats = evaluate_agent(agent, env, args.episodes)
    
    # Print statistics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of episodes: {args.episodes}")
    print(f"Mean Reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"Max Reward: {stats['max_reward']:.2f}")
    print(f"Min Reward: {stats['min_reward']:.2f}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print("="*50)
    
    # Plot results
    plot_results(stats)
    
    env.close()

if __name__ == '__main__':
    main()