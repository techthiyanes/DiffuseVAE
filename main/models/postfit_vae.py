# Fits a VAE on a bunch of latent codes !
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


# Implementation of the Resnet-VAE using a ResNet backbone as encoder
# and Upsampling blocks as the decoder
class PostVAE(pl.LightningModule):
    def __init__(
        self,
        in_channels,
        latent_dim,
        hidden_dims,
        alpha=1.0,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.alpha = alpha
        self.lr = lr

        # Encoder architecture
        modules = []
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 256]

        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim, bias=False),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1], latent_dim)

        # Decoder Architecture
        modules = []
        in_channels = latent_dim
        for h_dim in reversed(self.hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim, bias=False),
                    nn.ReLU())
            )
            in_channels = h_dim
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        out = self.encoder(x)
        mu = self.fc_mu(out)
        logvar = self.fc_var(out)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, z):
        # Only sample during inference
        decoder_out = self.decode(z)
        return decoder_out

    def forward_recons(self, x):
        # For generating reconstructions during inference
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        return decoder_out

    def training_step(self, batch, batch_idx):
        x = batch

        # Encoder
        mu, logvar = self.encode(x)

        # Reparameterization Trick
        z = self.reparameterize(mu, logvar)

        # Decoder
        decoder_out = self.decode(z)

        # Compute loss
        mse_loss = nn.MSELoss(reduction="sum")
        recons_loss = mse_loss(decoder_out, x)
        kl_loss = self.compute_kl(mu, logvar)
        self.log("Recons Loss", recons_loss, prog_bar=True)
        self.log("Kl Loss", kl_loss, prog_bar=True)

        total_loss = recons_loss + self.alpha * kl_loss
        self.log("Total Loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":

    vae = PostVAE(512, 256, None, alpha=1.0, lr=1e-4)

    sample = torch.randn(1, 512)
    out = vae.forward_recons(sample)
    print(out.shape)
