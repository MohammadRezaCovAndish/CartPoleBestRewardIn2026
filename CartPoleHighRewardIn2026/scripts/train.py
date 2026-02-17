#!/usr/bin/env python3
"""
Training script for CartPole DQN
"""

import os
import sys
import yaml
import argparse
from tqdm import tqdm

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import gymnasium as gym
import torch

# Now these imports will work
from agents.dqn_agent import DQNAgent
from utils.logger import Logger
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(config_path):
    """Main training function"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = gym.make(
        config['environment']['name'],
        max_episode_steps=config['environment']['max_episode_steps']
    )
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent
    agent = DQNAgent(state_dim, action_dim, config, device)
    
    # Create logger
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        use_tensorboard=config['logging']['tensorboard']
    )
    
    # Training loop
    print("\nStarting training...")
    
    for episode in tqdm(range(config['training']['total_episodes']), desc="Training"):
        
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            action = agent.act(state)
            
            # Take step
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience
            agent.step(state, action, reward, next_state, done or truncated)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        # Update target network
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        agent.episodes = episode + 1
        
        # Log episode
        logger.log_episode(
            episode=episode + 1,
            reward=episode_reward,
            length=episode_length,
            epsilon=agent.epsilon
        )
        
        # Save checkpoint
        if (episode + 1) % config['logging']['save_frequency'] == 0:
            save_checkpoint(
                agent,
                episode=episode + 1,
                reward=episode_reward,
                checkpoint_dir=config['logging']['checkpoint_dir']
            )
    
    # Save final model
    final_model_path = os.path.join(config['logging']['checkpoint_dir'], 'final_model.pt')
    agent.save(final_model_path)
    
    # Close logger
    logger.close()
    env.close()
    
    print("\nTraining completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Logs saved to: {config['logging']['log_dir']}")

def main():
    parser = argparse.ArgumentParser(description='Train DQN on CartPole')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    train(args.config)

if __name__ == '__main__':
    main()