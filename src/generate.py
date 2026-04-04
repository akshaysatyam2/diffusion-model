import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.model import DDPM_UNet
from src.schedule import get_ddpm_schedule

class AkshayMNISTEngine:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Igniting MNIST Engine on {self.device}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.names = checkpoint['metadata']['classes']
        
        self.model = DDPM_UNet()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.schedule = get_ddpm_schedule(1000, device=self.device)

    def generate(self, target_digit):
        print(f"Chiseling a '{self.names[target_digit]}' over 1000 steps...")
        
        with torch.no_grad():
            x = torch.randn(1, 1, 32, 32).to(self.device)
            label = torch.tensor([target_digit]).to(self.device)
            
            for i in reversed(range(1000)):
                t = torch.tensor([i], device=self.device).long()
                predicted_noise = self.model(x, label, t)
                
                alpha_t = self.schedule["alphas"][t].view(-1, 1, 1, 1)
                alpha_bar_t = self.schedule["alphas_cumprod"][t].view(-1, 1, 1, 1)
                beta_t = self.schedule["betas"][t].view(-1, 1, 1, 1)
                
                noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
            
            img = x.squeeze().cpu().numpy()
            img = np.clip((img + 1) / 2, 0, 1) 
            
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            plt.figure(figsize=(4, 4))
            plt.imshow(img, cmap='gray') 
            plt.title(f"MNIST DDPM | Class: {self.names[target_digit].upper()}")
            plt.axis('off')
            plt.show()
