import os
import torch

def save_checkpoint(
    agent,
    episode: int,
    reward: float,
    checkpoint_dir: str,
    filename: str = None
):
    """Save training checkpoint"""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_ep{episode}_reward{reward:.2f}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'episode': episode,
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'steps': agent.steps,
        'reward': reward
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(agent, checkpoint_path: str, load_optimizer: bool = True):
    """Load training checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    
    agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    agent.target_network.load_state_dict(checkpoint['model_state_dict'])
    
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    agent.epsilon = checkpoint.get('epsilon', agent.epsilon_min)
    agent.steps = checkpoint.get('steps', 0)
    
    episode = checkpoint.get('episode', 0)
    reward = checkpoint.get('reward', 0)
    
    print(f"Checkpoint loaded from episode {episode} with reward {reward:.2f}")
    return episode, reward