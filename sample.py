import torch

from unet import Unet, GuidedUnet
from diffusion import GaussianDiffusion

from configs.configs import make_config
from utils import make_im_grid


@torch.inference_mode()
def sample(diffusion: GaussianDiffusion, unet: Unet, 
           noise: torch.Tensor, timesteps: int = 1000):
    
    device = noise.device
    x = noise

    def step(xt, t, noise):
        eps = unet(xt, t, torch.zeros_like(t))
        out = diffusion.p_mean_variance(xt, t, eps)

        x_prev = out['mean'] + torch.exp(0.5 * out['log_variance']) * noise

        return x_prev

    for idx in reversed(range(timesteps)):
        t = idx * torch.ones((x.shape[0],), dtype=torch.int64).to(device)

        if idx > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x = step(x, t, z)
    
    return x.clamp(-1, 1)


@torch.inference_mode()
def sample_guidance(diffusion: GaussianDiffusion, unet: GuidedUnet, w: float, 
                    noise: torch.Tensor, labels: torch.Tensor, timesteps: int = 1000):
    
    assert noise.shape[0] == labels.shape[0]

    device = noise.device
    x = noise

    def step(xt, t, labels, noise):
        score = unet(xt, t, torch.zeros_like(labels), uncond=True)
        conditioned_score = unet(xt, t, labels)
        eps = score + (conditioned_score - score) * w

        out = diffusion.p_mean_variance(xt, t, eps)
        x_prev = out['mean'] + torch.exp(0.5 * out['log_variance']) * noise
        return x_prev

    for idx in reversed(range(timesteps)):
        t = idx * torch.ones((x.shape[0],), dtype=torch.int64).to(device)

        if idx > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x = step(x, t, labels, z)
    
    return x.clamp(-1, 1)


label_idx = {'airplane': 0,
             'automobile': 1,
             'bird': 2,
             'cat': 3,
             'deer': 4,
             'dog': 5,
             'frog': 6,
             'horse': 7,
             'ship': 8,
             'truck': 9}


if __name__ == '__main__':
    device = 'cuda'

    config = make_config('guided_cifar')

    ckpt_path = ''
    checkpoint = torch.load(ckpt_path)
    
    if config['run']['guidance']:
        unet = GuidedUnet(**config['unet'])
        unet.load_state_dict(checkpoint)
        unet.eval().to(device)
    else:
        unet = Unet(**config['unet'])
        unet.load_state_dict(checkpoint)
        unet.eval().to(device)

    diffusion = GaussianDiffusion(scheduler_type=config['run']['scheduler_type'],
                                  timesteps=config['run']['timesteps'],
                                  device=device)

    label = label_idx['dog']
    w = 5

    n = 64
    shape = (n,) + (3, 32, 32)

    interior = torch.ones(n, dtype=torch.int64) * label
    labels = torch.repeat_interleave(interior, 1).to(device)

    noise = torch.randn(n, 3, 32, 32).to(device)

    if config['run']['guidance']:
        x = sample_guidance(diffusion, unet, w, noise, labels)
    else:
        x = sample(diffusion, unet, noise)
    
    images = make_im_grid(x, xy=(8, 8))