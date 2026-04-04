import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.generate import AkshayMNISTEngine

def test_all_digits():
    model_location = "best_mnist_ddpm.pt"
    if not os.path.exists(model_location):
        print(f"Error: Model checkpoint not found at {model_location}.")
        print("Please train the model first using: python main.py --mode train")
        return

    print("Loading model for testing...")
    ai = AkshayMNISTEngine(model_location)
    
    print("Generating digits 0 through 9...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Generated Digits (0-9) using MNIST DDPM", fontsize=16)
    
    for i in range(10):
        print(f"[{i+1}/10] Generating digit '{ai.names[i]}'...")
        with torch.no_grad():
            x = torch.randn(1, 1, 32, 32).to(ai.device)
            label = torch.tensor([i]).to(ai.device)
            
            for step in reversed(range(1000)):
                t = torch.tensor([step], device=ai.device).long()
                predicted_noise = ai.model(x, label, t)
                
                alpha_t = ai.schedule["alphas"][t].view(-1, 1, 1, 1)
                alpha_bar_t = ai.schedule["alphas_cumprod"][t].view(-1, 1, 1, 1)
                beta_t = ai.schedule["betas"][t].view(-1, 1, 1, 1)
                
                noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
            
            img = x.squeeze().cpu().numpy()
            img = np.clip((img + 1) / 2, 0, 1) 
            
            # Upscale for better viewing in the grid
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
            
            # Plot
            ax = axes[i // 5, i % 5]
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Class: {ai.names[i].upper()}")
            ax.axis('off')
    
    plt.tight_layout()
    output_path = "test_results.png"
    plt.savefig(output_path)
    print(f"\n✅ Test complete! Grid saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    test_all_digits()
