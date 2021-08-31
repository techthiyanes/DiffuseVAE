import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vae import VAE as vae_nn


class VAE(pl.LightningModule):
    def __init__(
        self,
        enc_block_str,
        dec_block_str,
        enc_channel_str,
        dec_channel_str,
        alpha=1.0,
        lr=1e-4,
    ):
        super().__init__()
        self.enc_block_str = enc_block_str
        self.dec_block_str = dec_block_str
        self.enc_channel_str = enc_channel_str
        self.dec_channel_str = dec_channel_str
        self.alpha = alpha

        self.vae = vae_nn(
            enc_block_str=self.enc_block_str,
            dec_block_str=self.dec_block_str,
            enc_channel_str=self.enc_channel_str,
            dec_channel_str=self.dec_channel_str,
            alpha=self.alpha,
        )
        self.lr = lr

    def forward(self, z):
        return self.vae.sample(z)

    def training_step(self, batch, batch_idx):
        x = batch

        out, kl_loss = self.vae(x)
        mse_loss = nn.MSELoss(reduction="sum")
        recons_loss = mse_loss(out, x)
        total_loss = recons_loss + self.alpha * kl_loss
        self.log("Recons Loss", recons_loss, prog_bar=True)
        self.log("Kl Loss", kl_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
