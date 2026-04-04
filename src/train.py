import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from src.model import DDPM_UNet
from src.schedule import get_ddpm_schedule

def train_mnist_ddpm(epochs=1500, patience=150):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing V3.0 (MNIST Scale) Training on {device}...")
    
    # 1. OFFICIAL MNIST DOWNLOAD & TRANSFORM
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # Fix the 28x28 pooling math
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # 2. MASSIVE BATCH SIZE
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2) 
    print(f"Dataset Loaded: {len(dataset)} images.")
    
    model = DDPM_UNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    schedule = get_ddpm_schedule(timesteps=1000, device=device)
    sqrt_alphas_cumprod = schedule["sqrt_alphas_cumprod"]
    sqrt_one_minus_alphas_cumprod = schedule["sqrt_one_minus_alphas_cumprod"]

    best_loss = float('inf')
    epochs_no_improve = 0 

    for epoch in range(epochs):
        epoch_loss = 0.0 
        
        for step, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.shape[0]
            
            t = torch.randint(0, 1000, (batch_size,), device=device).long()
            noise = torch.randn_like(images).to(device)
            
            sqrt_alpha_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
            
            noisy_images = sqrt_alpha_t * images + sqrt_one_minus_alpha_t * noise
            
            optimizer.zero_grad()
            predicted_noise = model(noisy_images, labels, t)
            
            loss = loss_fn(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0 
            
            checkpoint = {
                'state_dict': model.state_dict(),
                'metadata': {
                    'creator': "Akshay Kumar",
                    'version': "3.0 - Full MNIST",
                    'architecture': "DDPM YOLO-U-Net (32x32)",
                    'classes': {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 
                                5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
                }
            }
            torch.save(checkpoint, 'best_mnist_ddpm.pt')
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\n🛑 Early stopping triggered!")
            break
            
    print("\n✅ Training Complete!")
