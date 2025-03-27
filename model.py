import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, t, z):
        return self.net(z)

class NeuralODEPKModel(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.encoder = nn.Linear(1, latent_dim)
        self.odefunc = ODEFunc(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, dose_amts, dose_times, obs_times):
        device = dose_amts.device
        z0 = self.encoder(dose_amts[0].unsqueeze(0))

        all_times = torch.cat([dose_times, obs_times]).unique().sort()[0]
        z_t = odeint(self.odefunc, z0, all_times)

        obs_idx = torch.tensor([torch.where(all_times == t)[0] for t in obs_times]).squeeze()
        z_obs = z_t[obs_idx]

        return self.decoder(z_obs)