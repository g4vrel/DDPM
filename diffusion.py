import math
import numpy as np

import torch

from einops import rearrange


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_betas(scheduler_type, timesteps):
    if scheduler_type == 'linear':
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, timesteps, dtype=np.float64
        )
    elif scheduler_type == 'cosine':
        return betas_for_alpha_bar(
            timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise ValueError


def ext(x):
    return rearrange(x, 'b -> b 1 1 1')


class GaussianDiffusion:
    def __init__(self, scheduler_type:str = 'linear', timesteps:int = 1000, device: str = 'cuda'):
        betas = get_betas(scheduler_type, timesteps)
        assert betas.dtype == np.float64, 'betas should be np.float64 for accuracy'
        self.device = device
        self.register_scheduler(betas)

    def register_scheduler(self, betas):
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = np.log(
            np.append(posterior_variance[1], posterior_variance[1:])
        )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - alphas_cumprod)
        )

        # fixed large
        model_variance = np.append(posterior_variance[1], betas[1:])
        model_log_variance = np.log(model_variance)

        scheduler = {'betas': betas,
                     'alphas': alphas,
                     'alphas_cumprod': alphas_cumprod,
                     'alphas_cumprod_prev': alphas_cumprod_prev,
                     'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
                     'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
                     'log_one_minus_alphas_cumprod': log_one_minus_alphas_cumprod,
                     'sqrt_recip_alphas_cumprod': sqrt_recip_alphas_cumprod,
                     'sqrt_recipm1_alphas_cumprod': sqrt_recipm1_alphas_cumprod,
                     'posterior_variance': posterior_variance,
                     'posterior_log_variance_clipped': posterior_log_variance_clipped,
                     'posterior_mean_coef1': posterior_mean_coef1,
                     'posterior_mean_coef2': posterior_mean_coef2,
                     'model_variance': model_variance,
                     'model_log_variance': model_log_variance}
        
        for name, arr in scheduler.items():
            setattr(self, name, torch.from_numpy(arr).float().to(self.device)) # float64 -> float32


    def forward(self, x0, t, noise):
        return self.q_sample(x0, t, noise)


    def q_mean_variance(self, x0, t):
        mean = ext(self.sqrt_alphas_cumprod[t]) * x0
        variance = ext(1.0 - self.alphas_cumprod[t])
        log_variance = ext(self.log_one_minus_alphas_cumprod[t])
        
        return {'mean': mean,
                'variance': variance,
                'log_variance': log_variance}


    def q_posterior_mean_variance(self, x0, xt, t):
        mean = ext(self.posterior_mean_coef1[t]) * x0 + ext(self.posterior_mean_coef2[t]) * xt
        variance = ext(self.posterior_variance[t])
        log_variance = ext(self.posterior_log_variance_clipped[t])
        
        return {'mean': mean,
                'variance': variance,
                'log_variance': log_variance}


    def q_sample(self, x0, t, noise):
        assert x0.shape == noise.shape
        
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        
        xt = ext(sqrt_alphas_cumprod) * x0 + ext(sqrt_one_minus_alphas_cumprod) * noise
        
        return xt


    def predict_xstart_from_eps(self, xt, t, eps):
        assert xt.shape == eps.shape
        
        return ext(self.sqrt_recip_alphas_cumprod[t]) * xt - ext(self.sqrt_recipm1_alphas_cumprod[t]) * eps
        

    def p_mean_variance(self, xt, t, eps, clip_denoised=True):
        assert xt.shape == eps.shape
        assert xt.shape[0] == t.shape[0]
        
        maybe_clip = lambda t: t.clamp(-1, 1) if clip_denoised else t

        pred_x0 = maybe_clip(self.predict_xstart_from_eps(xt, t, eps))
        
        mean = self.q_posterior_mean_variance(pred_x0, xt, t)['mean']
        variance = ext(self.model_variance[t])
        log_variance = ext(self.model_log_variance[t])
        
        return {'mean': mean,
                'variance': variance,
                'log_variance': log_variance}


    def p_sample(self, denoiser, xt, t, noise):
        assert xt.shape == noise.shape
        
        eps = denoiser(xt, t)
        out = self.p_mean_variance(xt, t, eps)
        x_prev = out['mean'] + torch.exp(0.5 * out['log_variance']) * noise
        
        return {'x_prev': x_prev,
                'mean': out['mean'],
                'variance': out['variance'],
                'log_variance': out['log_variance']}