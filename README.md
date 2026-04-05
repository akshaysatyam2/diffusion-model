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

<p align="center">
  <img src="images/ddpm_overfit/Picture%201.png" width="200" />
  <img src="images/ddpm_overfit/Picture%202.png" width="200" />
  <img src="images/ddpm_overfit/Picture%203.png" width="200" />
</p>
<p align="center"><em>(See the `images/ddpm_overfit` directory for more examples)</em></p>

### Generated from Final Model (MNIST)
These images show the final generated digits:

<p align="center">
  <img src="images/mnist/download%20(3).png" width="150" />
  <img src="images/mnist/download%20(4).png" width="150" />
  <img src="images/mnist/download%20(5).png" width="150" />
  <img src="images/mnist/download%20(7).png" width="150" />
  <img src="images/mnist/download%20(8).png" width="150" />
</p>
<p align="center"><em>(See the `images/mnist` directory for more examples)</em></p>

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

### Testing the Model
To test the model by generating a grid of all digits (0-9) and saving the result as `test_results.png`, run:
```bash
python test_model.py
```

<p align="center">
  <img src="test_results.png" width="800" alt="Generated digits grid (0-9)" />
</p>

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
