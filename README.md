# MNIST Diffusion Model (DDPM)

This repository contains a modular PyTorch implementation of a Denoising Diffusion Probabilistic Model (DDPM) trained on the MNIST dataset. The model generates 32x32 images of hand-drawn digits conditioned on a class label (0-9).

## Features
- Custom **U-Net Architecture** with GroupNorm and time/label embeddings.
- Full support for conditional generation of specific digits.
- Early stopping based on MSE loss.
- High-quality upscaled output (using OpenCV).

## Generated Images

### Generated from Initial Model (Overfitted)
These images show early results during training:
*(See the `images/ddpm_overfit` directory for more examples)*

### Generated from Final Model (MNIST)
These images show the final generated digits:
*(See the `images/mnist` directory for more examples)*

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:akshaysatyam2/diffusion-model.git
   cd diffusion-model
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To train the model from scratch, simply run:
```bash
python main.py --mode train --epochs 150
```
This will download the MNIST dataset to `./data`, train the U-Net, and save the best checkpoint as `best_mnist_ddpm.pt`.

### Generating Images
To generate a digit using a trained model, run:
```bash
python main.py --mode generate --digit 8
```
*(Replace `8` with any digit from `0-9`)*

## Architecture Summary
- **Input:** 32x32 image (MNIST images are resized).
- **Embeddings:** Sinusoidal time embeddings and learnable class embeddings are projected and added to the bottleneck.
- **Blocks:** Convolutional blocks with GroupNorm and LeakyReLU activations.
- **Loss:** Mean Squared Error (MSE) between actual noise and predicted noise.

## Author
Akshay Kumar
