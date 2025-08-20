# -*- coding: utf-8 -*-
"""Config and imports for Conditional GAN"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# Parameters
img_size = 28
num_classes = 2  # "circle" and "square"
latent_dim = 100
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
