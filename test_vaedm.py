import click
import logging
import math
import os
import torch

from pytorch_lightning.utilities.seed import seed_everything

from models.diffusion import (
    DDPM,
    UNetModel,
    SuperResModel,
)
from models.vaedm import VAEDM
from models.vae import VAE
from util import configure_device, save_as_images


logger = logging.getLogger(__name__)


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@click.command()
@click.argument("chkpt-path")
@click.option(
    "--enc-block-config",
    default="128x1,128d2,128t64,64x3,64d2,64t32,32x3,32d2,32t16,16x7,16d2,16t8,8x3,8d2,8t4,4x3,4d4,4t1,1x2",
)
@click.option(
    "--enc-channel-config", default="128:64,64:64,32:128,16:128,8:256,4:512,1:1024"
)
@click.option(
    "--dec-block-config",
    default="1x1,1u4,1t4,4x2,4u2,4t8,8x2,8u2,8t16,16x6,16u2,16t32,32x2,32u2,32t64,64x2,64u2,64t128,128x1",
)
@click.option(
    "--dec-channel-config", default="128:64,64:64,32:128,16:128,8:256,4:512,1:1024"
)
@click.option("--dim", default=128)
@click.option("--attn-resolutions", default="16,")
@click.option("--n-residual", default=2)
@click.option("--dim-mults", default="1,1,2,2,4,4")
@click.option("--dropout", default=0)
@click.option("--n-heads", default=1)
@click.option("--beta1", default=1e-4, type=float)
@click.option("--beta2", default=0.02, type=float)
@click.option("--n-timesteps", default=1000)
@click.option("--n-sample-steps", default=1000)
@click.option("--seed", default=0)
@click.option("--device", default="gpu:0")
@click.option("--image-size", default=128)
@click.option("--use-cond", default=True, type=bool)
@click.option("--n-samples", default=1)
@click.option("--save-path", default=os.getcwd())
def test(chkpt_path, **kwargs):
    # Set seed
    seed_everything(kwargs.get("seed"), workers=True)
    n_samples = kwargs.get("n_samples")
    batch_size = min(n_samples, 16)

    dev, _ = configure_device(kwargs.get("device"))

    # VAE model
    vae = VAE(
        enc_block_str=kwargs.get("enc_block_config"),
        dec_block_str=kwargs.get("dec_block_config"),
        enc_channel_str=kwargs.get("enc_channel_config"),
        dec_channel_str=kwargs.get("dec_channel_config"),
        alpha=1.0,
    ).to(dev)

    # DDPM Model
    attn_resolutions = __parse_str(kwargs.get("attn_resolutions"))
    dim_mults = __parse_str(kwargs.get("dim_mults"))

    # Use the superres model for conditional training
    decoder_cls = UNetModel if not kwargs.get("use_cond") else SuperResModel
    decoder = decoder_cls(
        in_channels=3,
        model_channels=kwargs.get("dim"),
        out_channels=3,
        num_res_blocks=kwargs.get("n_residual"),
        attention_resolutions=attn_resolutions,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=kwargs.get("dropout"),
        num_heads=kwargs.get("n_heads"),
    ).to(dev)

    ddpm = DDPM(
        decoder,
        beta_1=kwargs.get("beta1"),
        beta_2=kwargs.get("beta2"),
        T=kwargs.get("n_timesteps"),
    ).to(dev)

    # Combined model
    vaedm = VAEDM.load_from_checkpoint(
        chkpt_path,
        vae=vae,
        ddpm=ddpm,
        vae_lr=kwargs.get("vae_lr"),
        ddpm_lr=kwargs.get("ddpm_lr"),
    ).to(dev)

    decoder.eval()
    vae.eval()
    ddpm.eval()
    vaedm.eval()

    image_size = kwargs.get("image_size")
    n_steps = kwargs.get("n_sample_steps")
    ddpm_samples_list = []
    vae_samples_list = []
    for idx in range(math.ceil(n_samples / batch_size)):
        with torch.no_grad():
            # Sample from VAEDM
            x_t = torch.randn(batch_size, 3, image_size, image_size).to(dev)
            z = torch.randn(batch_size, 1024, 1, 1)
            vae_sample, ddpm_sample = vaedm(z_vae=z, z_ddpm=x_t, n_steps=n_steps)
            ddpm_samples_list.append(ddpm_sample.cpu())
            vae_samples_list.append(vae_sample.cpu())

    ddpm_cat_preds = torch.cat(ddpm_samples_list[:n_samples], dim=0)
    vae_cat_preds = torch.cat(vae_samples_list[:n_samples], dim=0)

    save_path = os.path.join(kwargs.get("save_path"), str(n_steps))
    os.makedirs(save_path, exist_ok=True)

    save_as_images(ddpm_cat_preds, file_name=os.path.join(save_path, "output_ddpm"))
    save_as_images(vae_cat_preds, file_name=os.path.join(save_path, "output_vae"))


if __name__ == "__main__":
    test()
