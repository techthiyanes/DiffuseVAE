import click
import copy
import math
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as T

from pytorch_lightning.utilities.seed import seed_everything
from models.diffusion import UNetModel, DDPM, DDPMWrapper, SuperResModel
from models.vae import VAE

from util import save_as_images, configure_device, normalize, get_dataset


def compare_samples(samples, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    ncols = len(samples)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)

    for idx, (caption, img) in enumerate(samples.items()):
        ax[idx].imshow(img.permute(1, 2, 0))
        ax[idx].set_title(caption)
        ax[idx].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=100, pad_inches=0)

    plt.close()


@click.group()
def cli():
    pass


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("ddpm-chkpt-path")
@click.argument('root')
@click.option("--device", default="gpu:1")
@click.option("--image-size", default=128)
@click.option("--truncation", default=1.0, type=float)
@click.option("--save-path", default=os.getcwd())
@click.option('--n-samples', default=1)
@click.option("--n-steps", default=1000)
@click.option('--reuse-epsilon', default=False, type=bool)
@click.option('--use-concat', default=False, type=bool)
@click.option('--compare', default=True, type=bool)
@click.option("--seed", default=0)
def generate_recons(vae_chkpt_path, ddpm_chkpt_path, root, **kwargs):
    seed_everything(kwargs.get('seed'))
    dev, _ = configure_device(kwargs.get('device'))
    image_size = kwargs.get('image_size')
    z_dim = kwargs.get('z_dim')
    n_steps = kwargs.get('n_steps')
    n_samples = kwargs.get('n_samples')

    transforms = None
    dataset = get_dataset('recons', root, transform=transforms)

    # VAE model
    vae = VAE.load_from_checkpoint(vae_chkpt_path).to(dev)
    vae.eval()

    # Superres Model
    decoder = SuperResModel if kwargs.get('use_concat') else UNetModel
    unet = decoder(
        3,
        128,
        3,
        num_res_blocks=2,
        attention_resolutions=[
            16,
        ],
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0,
        num_heads=1,
    ).to(dev)

    online_network = DDPM(
        unet, beta_1=1e-4, beta_2=0.02, T=1000, truncation=kwargs.get('truncation'), reuse_epsilon=kwargs.get('reuse_epsilon'),
    ).to(dev)
    target_network = copy.deepcopy(online_network).to(dev)
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        ddpm_chkpt_path,
        online_network=online_network,
        target_network=target_network,
    ).to(dev)
    ddpm_wrapper.eval()

    ddpm_samples_list = []
    vae_samples_list = []
    orig_samples_list = []

    for idx, (recons, img) in enumerate(dataset):
        if idx > n_samples - 1:
            break

        with torch.no_grad():
            recons = recons.unsqueeze(0).to(dev)
            x_t = recons + 1.0 * torch.randn_like(recons)
            ddpm_sample = ddpm_wrapper(x_t, cond=recons, n_steps=n_steps).cpu()

        ddpm_samples_list.append(ddpm_sample)
        vae_samples_list.append(recons.cpu())
        orig_samples_list.append(img.unsqueeze(0))

    ddpm_cat_preds = torch.cat(ddpm_samples_list, dim=0)
    vae_cat_preds = torch.cat(vae_samples_list, dim=0)
    orig_samples_list = torch.cat(orig_samples_list, dim=0)

    # Save reconstruction
    save_path = kwargs.get('save_path')
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)
    save_as_images(ddpm_cat_preds, file_name=os.path.join(save_path, "output"))

    # Save a comparison of all images
    if kwargs.get('compare'):
        compare_path = os.path.join(save_path, "compare")
        os.makedirs(compare_path, exist_ok=True)
        for idx, (ddpm_pred, vae_pred, img) in enumerate(zip(ddpm_cat_preds, vae_cat_preds, orig_samples_list)):
            samples = {
                'VAE': vae_pred,
                'DDPM': ddpm_pred,
                'Original': img
            }
            compare_samples(
                samples,
                save_path=os.path.join(compare_path, f"compare_{idx}.png"),
            )


@cli.command()
@click.argument("chkpt-path")
@click.option("--device", default="gpu:1")
@click.option("--num-samples", default=1)
@click.option("--image-size", default=128)
@click.option("--save-path", default=os.getcwd())
@click.option("--n-steps", default=1000)
def sample(
    chkpt_path,
    device="gpu:1",
    num_samples=1,
    image_size=128,
    n_steps=1000,
    save_path=os.getcwd(),
):
    # Samples from the unconditional DDPM model (as proposed in the original paper)
    seed_everything(0)
    # TODO: Update this method to work for cpus
    dev, _ = configure_device(device)

    # Model
    unet = UNetModel(
        3,
        128,
        3,
        num_res_blocks=2,
        attention_resolutions=[
            16,
        ],
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0,
        num_heads=1,
    )
    online_network = DDPM(unet)
    target_network = copy.deepcopy(online_network)
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        chkpt_path,
        online_network=online_network,
        target_network=target_network,
    ).to(dev)
    ddpm_wrapper.eval()

    batch_size = min(16, num_samples)

    ddpm_samples_list = []
    for idx in range(math.ceil(num_samples / batch_size)):
        with torch.no_grad():
            # Sample from DDPM
            x_t = torch.randn(batch_size, 3, image_size, image_size).to(dev)
            ddpm_sample = ddpm_wrapper(x_t, n_steps=n_steps).cpu()
            ddpm_samples_list.append(ddpm_sample)

    ddpm_cat_preds = torch.cat(ddpm_samples_list[:num_samples], dim=0)

    # Save the image and reconstructions as numpy arrays
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)

    # Save a comparison of all images
    save_as_images(ddpm_cat_preds, file_name=os.path.join(save_path, "output"))


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("ddpm-chkpt-path")
@click.option("--device", default="gpu:1")
@click.option("--num-samples", default=1)
@click.option("--image-size", default=128)
@click.option("--z-dim", default=1024)
@click.option("--truncation", default=1.0, type=float)
@click.option("--save-path", default=os.getcwd())
@click.option("--n-steps", default=1000)
@click.option("--compare", default=True, type=bool)
@click.option('--use-concat', default=False, type=bool)
@click.option('--temp', default=1.0, type=float)
@click.option("--seed", default=0)
def sample_cond(
    vae_chkpt_path,
    ddpm_chkpt_path,
    device="gpu:1",
    num_samples=1,
    image_size=128,
    z_dim=1024,
    truncation=1.0,
    n_steps=1000,
    save_path=os.getcwd(),
    compare=True,
    use_concat=False,
    seed=0,
    temp=1.0,
):
    # Samples from the conditional DDPM model
    seed_everything(seed)
    # TODO: Update this method to work for cpus
    dev, _ = configure_device(device)

    vae = VAE.load_from_checkpoint(vae_chkpt_path).to(dev)
    vae.eval()

    # Model
    decoder = SuperResModel if use_concat else UNetModel
    unet = decoder(
        3,
        128,
        3,
        num_res_blocks=2,
        attention_resolutions=[
            16,
        ],
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0,
        num_heads=1,
    ).to(dev)
    online_network = DDPM(
        unet, beta_1=1e-4, beta_2=0.02, T=1000, truncation=truncation
    ).to(dev)
    target_network = copy.deepcopy(online_network).to(dev)
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        ddpm_chkpt_path,
        online_network=online_network,
        target_network=target_network,
    ).to(dev)
    ddpm_wrapper.eval()

    batch_size = min(16, num_samples)

    ddpm_samples_list = []
    vae_samples_list = []
    for idx in range(math.ceil(num_samples / batch_size)):
        with torch.no_grad():
            # Sample from VAE
            z = torch.randn(batch_size, z_dim, 1, 1, device=dev)
            recons_ = vae(z)
            vae_samples_list.append(recons_.cpu())

            # Sample from DDPM (starting with the VAE reconstruction as the mean)
            recons = recons_
            x_t = (recons + temp * torch.randn_like(recons)).to(dev)
            ddpm_sample = ddpm_wrapper(x_t, cond=recons, n_steps=n_steps).cpu()
            ddpm_samples_list.append(ddpm_sample)

    ddpm_cat_preds = torch.cat(ddpm_samples_list[:num_samples], dim=0)
    vae_cat_preds = torch.cat(vae_samples_list[:num_samples], dim=0)

    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)

    save_as_images(ddpm_cat_preds, file_name=os.path.join(save_path, "output"))

    # Save a comparison of all images
    if compare:
        compare_path = os.path.join(save_path, "compare")
        os.makedirs(compare_path, exist_ok=True)
        for idx, (ddpm_pred, vae_pred) in enumerate(zip(ddpm_cat_preds, vae_cat_preds)):
            compare_samples(
                vae_pred,
                ddpm_pred,
                save_path=os.path.join(compare_path, f"compare_{idx}.png"),
            )

def plot_interpolations(interpolations, save_path=None, figsize=(10, 5)):
    N = len(interpolations)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)

    for i, inter in enumerate(interpolations):
        ax[i].imshow(inter.squeeze().permute(1, 2, 0))
        ax[i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def compare_interpolations(
    interpolations_1, interpolations_2, save_path=None, figsize=(10, 2)
):
    assert len(interpolations_1) == len(interpolations_2)
    N = len(interpolations_1)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=2, ncols=N, figsize=figsize)

    for i, (inter_1, inter_2) in enumerate(zip(interpolations_1, interpolations_2)):
        ax[0, i].imshow(inter_1.squeeze().permute(1, 2, 0))
        ax[0, i].axis("off")

        ax[1, i].imshow(inter_2.squeeze().permute(1, 2, 0))
        ax[1, i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("ddpm-chkpt-path")
@click.option("--device", default="gpu:1")
@click.option("--image-size", default=128)
@click.option("--z-dim", default=1024)
@click.option("--truncation", default=1.0, type=float)
@click.option("--save-path", default=os.getcwd())
@click.option("--n-steps", default=1000)
@click.option("--n-interpolate", default=10)
@click.option('--reuse-epsilon', default=False, type=bool)
@click.option('--use-concat', default=True, type=bool)
@click.option('--temp', default=1.0, type=float)
@click.option("--seed", default=0)
def interpolate_vae(vae_chkpt_path, ddpm_chkpt_path, **kwargs):
    seed_everything(kwargs.get('seed'))
    dev, _ = configure_device(kwargs.get('device'))
    image_size = kwargs.get('image_size')
    z_dim = kwargs.get('z_dim')
    n_steps = kwargs.get('n_steps')

    # Lambdas for interpolation
    lam = torch.linspace(0, 1.0, steps=kwargs.get('n_interpolate'), device=dev)

    # VAE model
    vae = VAE.load_from_checkpoint(vae_chkpt_path).to(dev)
    vae.eval()

    # Superres Model
    decoder = SuperResModel if kwargs.get('use_concat') else UNetModel
    unet = decoder(
        3,
        128,
        3,
        num_res_blocks=2,
        attention_resolutions=[
            16,
        ],
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0,
        num_heads=1,
    ).to(dev)

    online_network = DDPM(
        unet, beta_1=1e-4, beta_2=0.02, T=1000, truncation=kwargs.get('truncation'), reuse_epsilon=kwargs.get('reuse_epsilon'),
    ).to(dev)
    target_network = copy.deepcopy(online_network).to(dev)
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        ddpm_chkpt_path,
        online_network=online_network,
        target_network=target_network,
    ).to(dev)
    ddpm_wrapper.eval()

    ddpm_samples_list = []
    vae_samples_list = []

    with torch.no_grad():
        # Interpolate in the VAE latent space
        z_1 = torch.randn(1, z_dim, 1, 1, device=dev)
        z_2 = torch.randn(1, z_dim, 1, 1, device=dev)

        for idx, l in enumerate(lam):
            # Sample from VAE
            z_inter = z_1 * l + z_2 * (1 - l)
            recons_inter = vae(z_inter)

            vae_samples_list.append(recons_inter.cpu())

            # Sample from DDPM
            x_t = (recons_inter + kwargs,get('temp') * torch.randn_like(recons_inter)).to(dev)
            ddpm_sample = ddpm_wrapper(x_t, cond=recons_inter, n_steps=n_steps).cpu()
            ddpm_samples_list.append(normalize(ddpm_sample))

    # Compare
    save_path = kwargs.get('save_path')
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)
    compare_interpolations(
        ddpm_samples_list,
        vae_samples_list,
        save_path=os.path.join(save_path, "inter_compare.png"),
    )


@cli.command()
@click.argument("vae-chkpt-path")
@click.argument("ddpm-chkpt-path")
@click.option("--device", default="gpu:1")
@click.option("--image-size", default=128)
@click.option("--z-dim", default=1024)
@click.option("--truncation", default=1.0, type=float)
@click.option("--save-path", default=os.getcwd())
@click.option("--n-steps", default=1000)
@click.option("--n-interpolate", default=10)
@click.option('--reuse-epsilon', default=False, type=bool)
@click.option('--temp', default=1.0, type=float)
@click.option("--seed", default=0)
def interpolate_ddpm(vae_chkpt_path, ddpm_chkpt_path, **kwargs):
    seed_everything(kwargs.get('seed'))
    dev, _ = configure_device(kwargs.get('device'))
    image_size = kwargs.get('image_size')
    z_dim = kwargs.get('z_dim')
    n_steps = kwargs.get('n_steps')

    # Lambdas for interpolation
    lam = torch.linspace(0, 1.0, steps=kwargs.get('n_interpolate'), device=dev)

    # VAE model
    vae = VAE.load_from_checkpoint(vae_chkpt_path).to(dev)
    vae.eval()

    # Superres Model
    unet = SuperResModel(
        3,
        128,
        3,
        num_res_blocks=2,
        attention_resolutions=[
            16,
        ],
        channel_mult=(1, 1, 2, 2, 4, 4),
        dropout=0,
        num_heads=1,
    ).to(dev)

    online_network = DDPM(
        unet, beta_1=1e-4, beta_2=0.02, T=1000, truncation=kwargs.get('truncation'), reuse_epsilon=kwargs.get('reuse_epsilon'),
    ).to(dev)
    target_network = copy.deepcopy(online_network).to(dev)
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        ddpm_chkpt_path,
        online_network=online_network,
        target_network=target_network,
    ).to(dev)
    ddpm_wrapper.eval()

    ddpm_samples_list = []
    vae_samples_list = []

    with torch.no_grad():
        # Interpolate in the DDPM latent space
        z_1 = torch.randn(1, z_dim, 1, 1, device=dev)
        recons_inter = vae(z_1)

        x_t1 = recons_inter + kwargs,get('temp') * torch.randn(1, 3, image_size, image_size, device=dev)
        x_t2 = recons_inter + kwargs,get('temp') * torch.randn(1, 3, image_size, image_size, device=dev)

        for idx, l in enumerate(lam):
            # Sample from DDPM
            x_t_inter = x_t1 * l + x_t2 * (1 - l)
            ddpm_sample = ddpm_wrapper(x_t_inter, cond=recons_inter, n_steps=n_steps).cpu()
            ddpm_samples_list.append(normalize(ddpm_sample))
            vae_samples_list.append(recons_inter.cpu())

    # Compare
    save_path = kwargs.get('save_path')
    save_path = os.path.join(save_path, str(n_steps))
    os.makedirs(save_path, exist_ok=True)
    plot_interpolations(
        ddpm_samples_list, save_path=os.path.join(save_path, "inter_plot.png")
    )


if __name__ == "__main__":
    cli()
