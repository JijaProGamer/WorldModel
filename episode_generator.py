import gymnasium as gym
import torch
import json
import os
import uuid
import numpy as np

def main():
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    
    action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
    action_high = torch.tensor(env.action_space.high, dtype=torch.float32)

    os.makedirs('episodes', exist_ok=True)

    for _ in range(100_000):
        episode_data = []
        obs, info = env.reset()
        done = False

        while not done:            
            action_tensor = action_low + torch.rand(env.action_space.shape) * (action_high - action_low)
            action = action_tensor.numpy()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_data.append([obs.tolist(), action.tolist(), reward, terminated])
            
        episode_filename = str(uuid.uuid4()) + '.json'
        episode_filepath = os.path.join('episodes', episode_filename)

        with open(episode_filepath, 'w') as f:
            json.dump(episode_data, f)

    env.close()

if __name__ == "__main__":
    main()
