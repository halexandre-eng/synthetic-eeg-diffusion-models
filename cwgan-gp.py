"""
Synthetic EEG Channel Generation using Conditional WGAN-GP
This script trains a conditional WGAN-GP model to generate EEG channels
based on adjacent input channels. Includes training loop and synthetic signal generation.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gradient penalty function
def gradient_penalty(discriminator, real_data, fake_data, cond, lambda_gp=10):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, device=device)
    epsilon = epsilon.expand_as(real_data)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated = interpolated.detach().requires_grad_(True)

    prob_interpolated = discriminator(interpolated, cond)

    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

# Conditional Generator
class Generator(nn.Module):
    def __init__(self, z_dim, cond_dim, output_dim, n_features=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim + cond_dim, n_features * 4),
            nn.ReLU(),
            nn.Linear(n_features * 4, n_features * 8),
            nn.ReLU(),
            nn.Linear(n_features * 8, output_dim)
        )

    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.model(x)

# Conditional Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim, cond_dim, n_features=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + cond_dim, n_features * 8),
            nn.ReLU(),
            nn.Linear(n_features * 8, n_features * 4),
            nn.ReLU(),
            nn.Linear(n_features * 4, 1)
        )

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        return self.model(x)

# Training function for WGAN-GP
def train_wgan_gp(generator, discriminator, dataloader, num_epochs=100, z_dim=100, lambda_gp=10):
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    for epoch in range(num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        for batch_inputs, batch_targets in tqdm(dataloader, leave=False):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            batch_size = batch_inputs.size(0)
            cond = batch_inputs.reshape(batch_size, -1)
            real_data = batch_targets.reshape(batch_size, -1)

            # Train Discriminator
            for _ in range(5):
                z = torch.randn(batch_size, z_dim).to(device)
                fake_data = generator(z, cond)

                real_validity = discriminator(real_data, cond)
                fake_validity = discriminator(fake_data.detach(), cond)
                gp = gradient_penalty(discriminator, real_data, fake_data.detach(), cond, lambda_gp)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gp

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()
                epoch_d_loss += d_loss.item()

            # Train Generator
            z = torch.randn(batch_size, z_dim).to(device)
            fake_data = generator(z, cond)
            g_loss = -torch.mean(discriminator(fake_data, cond))

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
            epoch_g_loss += g_loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {epoch_d_loss / len(dataloader):.4f}, G Loss: {epoch_g_loss / len(dataloader):.4f}")

# Generate synthetic EEG samples
def generate_synthetic_data(generator, conditioning_inputs, z_dim=100):
    generator.eval()
    with torch.no_grad():
        batch_size = conditioning_inputs.size(0)
        z = torch.randn(batch_size, z_dim).to(device)
        synthetic_data = generator(z, conditioning_inputs)
    return synthetic_data.cpu().numpy()
