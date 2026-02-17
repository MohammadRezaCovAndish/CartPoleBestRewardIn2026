#!/usr/bin/env python3
"""
Test script for trained DQN model
"""

import os
import sys
import yaml
import argparse
import numpy as np

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

def test(config_path, model_path, episodes=10, render=False):
    """Test trained model"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create environment
    env = gym.make(
        config['environment']['name'],
        max_episode_steps=config['environment']['max_episode_steps'],
        render_mode='human' if render else None
    )
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_dim, action_dim, config, device)
    
    # Load model
    agent.load(model_path)
    
    # Test loop
    episode_rewards = []
    episode_lengths = []
    
    print(f"\nTesting model for {episodes} episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action (no exploration)
            action = agent.act(state, eval_mode=True)
            
            # Take step
            state, reward, done, truncated, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\nTest Results ({episodes} episodes):")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Length: {mean_length:.2f}")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='Test trained DQN model')
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
        default=10,
        help='Number of test episodes'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render the environment'
    )
    args = parser.parse_args()
    
    test(args.config, args.model, args.episodes, args.render)

if __name__ == '__main__':
    main()