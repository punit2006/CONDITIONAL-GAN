import numpy as np
import torch
from .config import img_size, num_classes, device

# Helper: Generate simple images
def generate_shape(label, size=img_size):
    img = np.zeros((size, size), dtype=np.float32)
    if label == 0:  # Circle
        rr, cc = np.ogrid[:size, :size]
        mask = (rr - size//2)**2 + (cc - size//2)**2 < (size//3)**2
        img[mask] = 1.0
    else:  # Square
        img[size//4:3*size//4, size//4:3*size//4] = 1.0
    return img

# Dataset batch
def get_batch(batch_size):
    labels = np.random.randint(0, num_classes, batch_size)
    imgs = np.stack([generate_shape(l) for l in labels])
    return torch.tensor(imgs).unsqueeze(1), torch.tensor(labels)

# One-hot encoding
def one_hot(labels, num_classes):
    return torch.eye(num_classes)[labels].to(device)
