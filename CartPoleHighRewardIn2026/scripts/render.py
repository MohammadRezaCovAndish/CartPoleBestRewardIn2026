#!/usr/bin/env python3
"""
Render trained model with visualization
"""

import os
import sys
import yaml
import argparse
import time

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

def render_episode(env, agent, delay=0.02):
    """Render a single episode"""
    state, _ = env.reset()
    episode_reward = 0
    done = False
    truncated = False
    
    while not (done or truncated):
        # Render
        env.render()
        
        # Select action
        action = agent.act(state, eval_mode=True)
        
        # Take step
        state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        
        # Small delay for visualization
        time.sleep(delay)
    
    return episode_reward

def main():
    parser = argparse.ArgumentParser(description='Render trained DQN model')
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
        default=5,
        help='Number of episodes to render'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.02,
        help='Delay between frames (seconds)'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create environment with rendering
    env = gym.make(
        config['environment']['name'],
        max_episode_steps=config['environment']['max_episode_steps'],
        render_mode='human'
    )
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_dim, action_dim, config, device)
    
    # Load model
    agent.load(args.model)
    
    print(f"\nRendering {args.episodes} episodes...")
    
    for episode in range(args.episodes):
        reward = render_episode(env, agent, args.delay)
        print(f"Episode {episode + 1}: Reward = {reward:.2f}")
    
    env.close()

if __name__ == '__main__':
    main()