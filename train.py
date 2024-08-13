import sys

from configs.configs import make_config
import utils

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from unet import Unet, GuidedUnet
from diffusion import GaussianDiffusion

import wandb

import torch
import torch.nn.functional as F

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LinearLR

import lightning as L


class DDPM(L.LightningModule):
    def __init__(self, denoiser, diffusion, config):
        super().__init__()
        self.save_hyperparameters(ignore=['denoiser', 'diffusion'])

        self.denoiser = denoiser
        self.diffusion = diffusion

        self.timesteps = config['run']['timesteps']
    

    def configure_optimizers(self):
        config = self.hparams['config']['train']
        lr = config['lr']
        warmup_steps = config['warmup_steps']

        if config['optimizer'] == 'adam':
            optimizer = Adam(self.denoiser.parameters(), lr=lr)
        elif config['optimizer'] == 'adamw':
            optimizer = AdamW(self.denoiser.parameters(), lr=lr)
        else:
            raise ValueError
        
        warm_sch = LinearLR(
            optimizer,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        return [optimizer], [warm_sch]


    def step(self, batch, batch_idx):
        x0, labels = batch[0], batch[1]
        
        t = torch.randint(0, self.timesteps, size=(x0.shape[0],), dtype=torch.long, device=x0.device)
        noise = torch.randn_like(x0)

        xt = self.diffusion.q_sample(x0, t, noise)
        eps = self.denoiser(xt, t, labels)
        
        loss = F.mse_loss(eps, noise, reduction='mean')
        
        return loss


    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train/mse_loss', loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('eval/mse_loss', loss, prog_bar=True)
        return loss


if __name__ == '__main__':
    wandb.login()

    file = None
    if len(sys.argv) == 2:
        file = sys.argv[1]
    
    config = make_config(file)

    device = 'cuda'

    if config['run']['guidance']:
        unet = GuidedUnet(**config['run']['unet'])
    else:
        unet = Unet(**config['run']['unet'])
    

    diffusion = GaussianDiffusion(scheduler_type=config['run']['scheduler_type'],
                                  timesteps=config['run']['timesteps'],
                                  device=device)

    model = DDPM(denoiser=unet, model=diffusion, config=config)

    train_loader, eval_loader = utils.get_loaders(config['train'])

    assert wandb.run is None

    logger = WandbLogger(name=file, project='TCC', log_model=False)
    logger.watch(model.denoiser, log_freq=config['train']['log_freq'])
    callbacks=[LearningRateMonitor('epoch')]
    torch.set_float32_matmul_precision('medium')
    
    trainer = L.Trainer(max_epochs=150,
                        logger=logger,
                        gradient_clip_algorithm='norm',
                        gradient_clip_val=config['train']['grad_clip_val'],
                        callbacks=callbacks,
                        benchmark=True,
                        precision=config['train']['precision'],
                        log_every_n_steps=150)
    
    trainer.fit(model, train_loader, eval_loader)

    wandb.finish()