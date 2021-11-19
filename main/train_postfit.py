import logging
import os
import pytorch_lightning as pl
import torchvision.transforms as T
import hydra
import numpy as np

from omegaconf import OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from models.postfit_vae import PostVAE
from datasets.latent import TensorDataset
from util import get_dataset, configure_device


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs")
def train(config):
    # Get config and setup
    config = config.dataset.post_vae
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)

    # Dataset
    root = config.data.root
    
    # Load latent codes
    z = np.load(root)
    dataset = TensorDataset(z)
    N = len(dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)

    # Model
    vae = PostVAE(
        in_channels=config.model.in_channels,
        latent_dim=config.model.latent_dim,
        hidden_dims=config.model.hidden_dims,
        alpha=config.training.alpha,
        lr=config.training.lr,
    )

    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path is not None:
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    results_dir = config.training.results_dir
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"postvae-{config.training.chkpt_prefix}"
        + "-{epoch:02d}-{train_loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    device = config.training.device
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        from pytorch_lightning.plugins import DDPPlugin

        train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8

    # Half precision training
    if config.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(vae, train_dataloader=loader)


if __name__ == "__main__":
    train()
