import torch
from torch.utils.data import Dataset
import json
import os

class EpisodeDataset(Dataset):
    def __init__(self, episodes_dir='episodes'):
        self.episodes_dir = episodes_dir
        self.episode_files = [f for f in os.listdir(episodes_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.episode_files)

    def __getitem__(self, idx):
        episode_filepath = os.path.join(self.episodes_dir, self.episode_files[idx])

        with open(episode_filepath, 'r') as f:
            episode_data = json.load(f)

        episode = []
        for step in episode_data:
            obs, action, reward, terminated = step
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            terminated_tensor = torch.tensor(terminated, dtype=torch.bool)
            episode.append((obs_tensor, action_tensor, reward_tensor, terminated_tensor))

        return episode
