import math
import torch
from torch import nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DDPM_UNet(nn.Module):
    def __init__(self):
        super(DDPM_UNet, self).__init__()
        
        self.label_embedding = nn.Embedding(10, 256)
        self.time_embedding = SinusoidalPositionEmbeddings(256)
        self.embedding_projector = nn.Linear(512, 512) 
        
        def gn_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_f), 
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_f, out_f, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_f),
                nn.LeakyReLU(0.1)
            )

        self.down1 = gn_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = gn_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = gn_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = gn_block(256, 512)
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decode3 = gn_block(512, 128) 
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.decode2 = gn_block(256, 64)  
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decode1 = gn_block(128, 64)  
        
        self.final_out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, noisy_image, label, t):
        x1 = self.down1(noisy_image); p1 = self.pool1(x1)               
        x2 = self.down2(p1); p2 = self.pool2(x2)               
        x3 = self.down3(p2); p3 = self.pool3(x3)               
        
        b = self.bottleneck(p3)           
        
        l_emb = self.label_embedding(label)
        t_emb = self.time_embedding(t)
        combined_emb = torch.cat([l_emb, t_emb], dim=-1) 
        projected_emb = self.embedding_projector(combined_emb).view(-1, 512, 1, 1)
        
        b = b + projected_emb 
        
        u1 = self.up1(b)                  
        d3 = self.decode3(torch.cat([u1, x3], dim=1))             
        u2 = self.up2(d3)                 
        d2 = self.decode2(torch.cat([u2, x2], dim=1))             
        u3 = self.up3(d2)                 
        d1 = self.decode1(torch.cat([u3, x1], dim=1))             
        
        return self.final_out(d1)
