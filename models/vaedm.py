import pytorch_lightning as pl
import torch.nn as nn
import torch


class VAEDM(pl.LightningModule):
    def __init__(
        self,
        vae,
        ddpm,
        vae_lr=1e-4,
        ddpm_lr=2e-5,
        vae_loss="l2",
        ddpm_loss="l1",
        alpha=1.0,
    ):
        super().__init__()
        assert vae_loss in ["l1", "l2"]
        assert ddpm_loss in ["l1", "l2"]
        self.vae = vae
        self.ddpm = ddpm

        self.vae_criterion = (
            nn.MSELoss(reduction="sum")
            if vae_loss == "l2"
            else nn.L1Loss(reduction="sum")
        )
        self.ddpm_criterion = (
            nn.MSELoss(reduction="sum")
            if ddpm_loss == "l2"
            else nn.L1Loss(reduction="sum")
        )
        self.vae_lr = vae_lr
        self.ddpm_lr = ddpm_lr
        self.alpha = alpha

    def forward(self, z_vae, z_ddpm, n_steps=None):
        # Sample from the VAE
        vae_recons = self.vae.sample(z_vae)

        # Sample from the DDPM model
        ddpm_recons = self.ddpm.sample(z_ddpm, cond=vae_recons, n_steps=n_steps)
        return vae_recons, ddpm_recons

    def training_step(self, batch, batch_idx):
        x = batch

        # Forward through the vae
        vae_recons, kl_loss = self.vae(x)

        # Forward pass through DDPM
        # Sample timepoints
        t = torch.randint(0, self.ddpm.T, size=(x.size(0),), device=self.device)

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise
        eps_pred = self.ddpm(x, eps, t, low_res=vae_recons)

        # Compute loss
        recons_loss = self.vae_criterion(vae_recons, x)
        ddpm_loss = self.ddpm_criterion(eps, eps_pred)
        total_loss = recons_loss + self.alpha * kl_loss + ddpm_loss
        self.log("Recons", recons_loss, prog_bar=True)
        self.log("KL", kl_loss, prog_bar=True)
        self.log("DDPM", ddpm_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.vae.parameters(), "lr": self.vae_lr},
                {"params": self.ddpm.parameters(), "lr": self.ddpm_lr},
            ],
        )
        return optimizer
