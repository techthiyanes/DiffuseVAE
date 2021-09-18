import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.diffusion.unet import Unet

from tqdm import tqdm


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDPM(nn.Module):
    def __init__(
        self,
        decoder,
        beta_1=1e-4,
        beta_2=0.02,
        T=1000,
        truncation=1.0,
        reuse_epsilon=False,
    ):
        super().__init__()
        self.decoder = decoder
        self.T = T
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.truncation = truncation
        self.reuse_epsilon = reuse_epsilon
        self.epsilon = None

        # Flag to keep track of device settings
        self.setup_consts = False

    def setup_precomputed_const(self, dev):
        # Main
        self.betas = torch.linspace(self.beta_1, self.beta_2, steps=self.T, device=dev)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_shifted = torch.cat(
            [torch.tensor([1.0], device=dev), self.alpha_bar[:-1]]
        )

        # Posterior covariance of the forward process
        self.post_variance = (
            (self.truncation ** 2)
            * self.betas
            * (1.0 - self.alpha_bar_shifted)
            / (1.0 - self.alpha_bar)
        )

        # Auxillary consts
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.minus_sqrt_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.post_coeff_1 = (
            torch.sqrt(self.alpha_bar_shifted) * self.betas / (1 - self.alpha_bar)
        )
        self.post_coeff_2 = (
            self.sqrt_alphas * (1 - self.alpha_bar_shifted) / (1 - self.alpha_bar)
        )
        self.post_coeff_3 = 1 - self.post_coeff_2

    def get_posterior_mean_covariance(self, x_t, t, clip_denoised=True, cond=None):
        t_ = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
        x_hat = 0 if cond is None else cond
        # Generate the reconstruction from x_t
        x_recons = (
            x_t
            - cond
            - self.decoder(x_t, t_, low_res=cond)
            * extract(self.minus_sqrt_alpha_bar, t_, x_t.shape)
        ) / extract(self.sqrt_alpha_bar, t_, x_t.shape)

        # Clip
        if clip_denoised:
            x_recons.clamp_(0, 1.0)

        # Compute posterior mean from the reconstruction
        post_mean = (
            extract(self.post_coeff_1, t_, x_t.shape) * x_recons
            + extract(self.post_coeff_2, t_, x_t.shape) * x_t
            + extract(self.post_coeff_3, t_, x_t.shape) * x_hat
        )
        post_variance = extract(self.post_variance, t_, x_t.shape)
        return post_mean, post_variance

    def sample(self, x_t, cond=None, n_steps=None, checkpoints=[]):
        if self.reuse_epsilon and self.epsilon is None:
            _, C, H, W = x_t.shape
            self.epsilon = torch.randn(self.T, C, H, W, device=x_t.device)

        # The sampling process goes here!
        x = x_t
        sample_dict = {}

        # Set device
        dev = x_t.device
        if not self.setup_consts:
            self.setup_precomputed_const(dev)
            self.setup_consts = True

        num_steps = self.T if n_steps is None else n_steps
        checkpoints = [num_steps] if checkpoints == [] else checkpoints
        for idx, t in enumerate(reversed(range(0, num_steps))):
            z = (
                torch.randn_like(x_t)
                if not self.reuse_epsilon
                else self.epsilon[t, :, :, :]
            )
            post_mean, post_variance = self.get_posterior_mean_covariance(
                x,
                t,
                cond=cond,
            )
            # Langevin step!
            x = post_mean + torch.sqrt(post_variance) * z

            # Add results
            if idx + 1 in checkpoints:
                sample_dict[str(idx + 1)] = x
        return sample_dict

    def compute_noisy_input(self, x_start, eps, t, low_res=None):
        assert eps.shape == x_start.shape
        x_recons = 0 if low_res is None else low_res
        # Samples the noisy input x_t ~ N(x_t|x_0) in the forward process
        return (
            x_start * extract(self.sqrt_alpha_bar, t, x_start.shape)
            + x_recons
            + self.truncation
            * eps
            * extract(self.minus_sqrt_alpha_bar, t, x_start.shape)
        )

    def forward(self, x, eps, t, low_res=None):
        if not self.setup_consts:
            self.setup_precomputed_const(x.device)
            self.setup_consts = True

        # Predict noise
        x_t = self.compute_noisy_input(x, eps, t, low_res=low_res)
        return self.decoder(x_t, t, low_res=low_res)


if __name__ == "__main__":
    decoder = Unet(64)
    ddpm = DDPM(decoder)
    t = torch.randint(0, 1000, size=(4,))
    sample = torch.randn(4, 3, 128, 128)
    loss = ddpm(sample, torch.randn_like(sample), t)
    print(loss)

    # Test sampling
    x_t = torch.randn(4, 3, 128, 128)
    samples = ddpm.sample(x_t)
    print(samples.shape)
