import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from tqdm import tqdm


from episodes_dataloader import EpisodeDataset
from WorldModel import Autoencoder

if __name__ == "__main__":
    input_dim = 3
    action_dim = 1

    hidden_dim = 64
    encoder_layers = 3
    decoder_layers = 3
    batch_size = 32
    num_epochs = 1000
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EpisodeDataset(episodes_dir="episodes")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_dim, action_dim, hidden_dim, encoder_layers, decoder_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch in progress_bar:
            optimizer.zero_grad()

            batch_obs = torch.stack([sample[0] for sample in batch])[:-1].to(device)
            batch_actions = torch.stack([sample[1] for sample in batch])[:-1].to(device)

            next_obs = [batch[i + 1][0] for i in range(len(batch) - 1)]
            batch_next_obs = torch.stack(next_obs).to(device)

            predicted_next_obs = model(batch_obs, batch_actions)

            loss = criterion(predicted_next_obs, batch_next_obs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
