import torch.nn as nn
from .config import img_size, latent_dim, num_classes

# === Basic Generator ===
class GeneratorBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Linear(num_classes, 16)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 16, 128),
            nn.ReLU(),
            nn.Linear(128, img_size*img_size),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([noise, label_input], 1)
        img = self.model(x)
        return img.view(-1, 1, img_size, img_size)

# === Basic Discriminator ===
class DiscriminatorBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Linear(num_classes, 16)
        self.model = nn.Sequential(
            nn.Linear(img_size*img_size + 16, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([img.view(img.size(0), -1), label_input], 1)
        validity = self.model(x)
        return validity

# === Advanced Generator ===
class GeneratorAdvanced(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_size*img_size),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([noise, label_input], 1)
        img = self.model(x)
        return img.view(-1, 1, img_size, img_size)

# === Advanced Discriminator ===
class DiscriminatorAdvanced(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(img_size*img_size + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([img.view(img.size(0), -1), label_input], 1)
        validity = self.model(x)
        return validity
