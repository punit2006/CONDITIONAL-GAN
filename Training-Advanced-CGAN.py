import torch
import torch.nn as nn
import torch.optim as optim
from .config import device, latent_dim, batch_size, num_classes
from .dataset import get_batch
from .models import GeneratorAdvanced, DiscriminatorAdvanced

def train_advanced(epochs=10000):
    generator = GeneratorAdvanced().to(device)
    discriminator = DiscriminatorAdvanced().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    adversarial_loss = nn.BCELoss()

    for epoch in range(epochs):
        real_imgs, labels = get_batch(batch_size)
        real_imgs, labels = real_imgs.to(device), labels.to(device)
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)

        z = torch.randn(batch_size, latent_dim, device=device)
        gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
        gen_imgs = generator(z, gen_labels)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        gen_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
        gen_loss.backward()
        optimizer_G.step()

        if epoch % 1000 == 0:
            print(f"[Advanced] Epoch {epoch} D loss: {d_loss.item():.4f} G loss: {gen_loss.item():.4f}")
