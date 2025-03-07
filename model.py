import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, latent_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 50), nn.Tanh(), nn.Linear(50, latent_dim))

    def forward(self, t, z):
        return self.net(z)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 50), nn.ReLU(), nn.Linear(50, latent_dim))

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 50), nn.ReLU(), nn.Linear(50, output_dim))

    def forward(self, z):
        return self.net(z)

class NeuralODEPKModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(NeuralODEPKModel, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.odefunc = ODEFunc(latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)

    def forward(self, x0, t):
        z0 = self.encoder(x0)
        z_t = odeint(self.odefunc, z0, t)
        return self.decoder(z_t)