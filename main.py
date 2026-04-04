import os
import argparse
from src.train import train_mnist_ddpm
from src.generate import AkshayMNISTEngine

def main():
    parser = argparse.ArgumentParser(description="DDPM MNIST Diffusion Model")
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='generate', help='Mode to run: train or generate')
    parser.add_argument('--digit', type=int, default=8, help='Digit to generate (0-9)')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mnist_ddpm(epochs=args.epochs, patience=15)
    elif args.mode == 'generate':
        model_location = "best_mnist_ddpm.pt"
        if os.path.exists(model_location):
            ai = AkshayMNISTEngine(model_location)
            ai.generate(args.digit)
        else:
            print(f"Model checkpoint not found at {model_location}. Please train the model first.")

if __name__ == "__main__":
    main()
