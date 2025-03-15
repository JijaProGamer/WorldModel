import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Encoder, self).__init__()

        self.layers = []

        input_dim = input_dim
        for i in range(1, layers+1):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if i < layers:
                self.layers.append(nn.GELU())
                #self.layers.append(nn.RMSNorm(hidden_dim))
                self.layers.append(nn.LayerNorm(hidden_dim))
            
            input_dim = hidden_dim

        self.network = nn.Sequential(*self.layers)
    
    def forward(self, x):
        encoded = self.network(x)
        return encoded
    
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, layers):
        super(Decoder, self).__init__()

        self.layers = []

        out_dims = hidden_dim
        for i in range(1, layers+1):
            if i == layers:
                out_dims = output_dim


            self.layers.append(nn.Linear(hidden_dim, out_dims))
            if i < layers:
                self.layers.append(nn.GELU())
                #self.layers.append(nn.RMSNorm(hidden_dim))
                self.layers.append(nn.LayerNorm(hidden_dim))

        self.network = nn.Sequential(*self.layers)
    
    def forward(self, x):
        encoded = self.network(x)
        return encoded
    
class Autoencoder(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, encoder_layers, decoder_layers):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(input_dim + action_dim, hidden_dim, encoder_layers)
        self.decoder = Decoder(hidden_dim, input_dim, decoder_layers)
    
    """def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded"""
    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded