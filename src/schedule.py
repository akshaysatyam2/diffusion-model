import torch

def get_ddpm_schedule(timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return {
        "betas": betas, "alphas": alphas, "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod)
    }
