"""
Synthetic EEG Channel Generation using Denoising Diffusion Probabilistic Models (DDPM)
This script trains a conditional diffusion model based on a U-Net architecture to generate EEG channels
based on adjacent input channels.

"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Beta scheduler
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# Time embedding module
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# Conditional 1D U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_feat=64, time_emb_dim=256):
        super(UNet, self).__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels + time_emb_dim, n_feat, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_feat),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_feat, n_feat * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_feat * 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(n_feat * 2, n_feat, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_feat),
            nn.ReLU()
        )
        self.conv4 = nn.ConvTranspose1d(n_feat, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t).unsqueeze(-1).repeat(1, 1, x.size(-1))
        x = torch.cat([x, t_emb], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

# Training function
def train_diffusion_model(model, optimizer, scheduler, dataloader, timesteps, betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (batch_inputs, batch_targets) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_size = batch_inputs.size(0)
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(batch_targets)
            sqrt_acp_t = sqrt_alphas_cumprod[t].view(batch_size, 1, 1)
            sqrt_om_acp_t = sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1)
            x_t = sqrt_acp_t * batch_targets + sqrt_om_acp_t * noise
            model_input = torch.cat([batch_inputs, x_t], dim=1)
            optimizer.zero_grad()
            noise_pred = model(model_input, t)
            loss = nn.MSELoss()(noise_pred, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
        scheduler.step()

# Sampling function
def sample_from_model(model, conditioning_inputs, alphas, alphas_cumprod, betas, sqrt_one_minus_alphas_cumprod, shape):
    model.eval()
    batch_size = shape[0]
    seq_length = shape[2]
    x = torch.randn((batch_size, 1, seq_length), device=device)
    with torch.no_grad():
        for t in reversed(range(len(betas))):
            t_batch = torch.tensor([t]*batch_size, device=device).long()
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            model_input = torch.cat([conditioning_inputs, x], dim=1)
            noise_pred = model(model_input, t_batch)
            x = (1 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred)
    return x
