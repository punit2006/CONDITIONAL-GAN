## CGAN: Conditional GAN for Simple Shapes (Circles & Squares)

This repository implements a Conditional Generative Adversarial Network (CGAN) in PyTorch to generate simple images of circles and squares based on provided class labels. The project includes two models and compares their results visually using generated samples.

**Google Colab Notebook:**  

👉 https://colab.research.google.com/drive/1krLKX8JubAF2zlrOnY-JnjXX6y5x0wms?usp=sharing

***

### Overview

Conditional GANs (CGANs) allow the generator to create images conditioned on class labels, enabling targeted generation—for example, generating either circles or squares. This repository contains two CGAN implementations that vary in architecture.

***

### Features

- **Two CGAN Models:** Each model uses different PyTorch architectures.
- **Shapes Supported:** Circle and Square images (28x28 px).
- **Training Visualization:** See generator outputs during training.
- **Easy to Run:** Works on CPU and CUDA.

***

### Setup

1. Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/CGAN-shapes.git
cd CGAN-shapes
pip install torch torchvision matplotlib numpy
```

2. Or run directly in Google Colab : https://colab.research.google.com/drive/1krLKX8JubAF2zlrOnY-JnjXX6y5x0wms?usp=sharing

***

### Usage

Run either of the included scripts to train a CGAN on circles and squares. Training logs and output images plot sample generations over epochs.

```bash
python 2_cgan.py
```

***

### Model Architectures

- **Model 1:** Simple linear layers with one-hot encoded labels.
- **Model 2:** Deeper architecture with embedding and BatchNorm layers for improved sample quality.

***

### Results

Below are sample outputs at various stages of training, comparing both architectures.

#### Model 1 - Output Samples
![Model 1 Output Samples (Epoch 9000)](attachment_id="attached 2 - Output Samples
![Model 2 Output Samples (Epoch 900)](attachment_id="attachedation:**
- The first image shows results from Model 1 after 9,000 epochs: clearer geometric shapes.
- The second image shows results from Model 2 after 900 epochs: initial rough features, shapes are less clean.

***

### File Structure

- `2_cgan.py`: Main training and generation script with both model implementations.

***

### Citations & References

- [PyTorch Documentation](https://pytorch.org/)
- [Goodfellow et al., GAN Paper](https://arxiv.org/abs/1406.2661)

***

### License

This project is released under the MIT License.

***

### Contact

For questions and contributions, open an issue or pull request on this repository.

