import click
import logging
import os
import pytorch_lightning as pl
import torchvision.transforms as T

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.seed import seed_everything

from models.vae import VAE
from util import configure_device, get_dataset


logger = logging.getLogger(__name__)


@click.command()
@click.argument("root")
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
@click.option("--batch-size", default=16)
@click.option("--epochs", default=1000)
@click.option("--image-size", default=128)
@click.option("--workers", default=8)
@click.option("--lr", default=1e-4)
@click.option("--log-step", default=1)
@click.option("--device", default="gpu:0")
@click.option("--dataset", default="celeba-hq")
@click.option("--chkpt-interval", default=1)
@click.option("--optimizer", default="Adam")
@click.option("--restore-path", default=None)
@click.option("--results-dir", default=os.getcwd())
@click.option("--wandb-run-name", default="dummy")
@click.option("--fp16", default=False)
@click.option("--seed", default=0)
def train(root, **kwargs):
    # Set seed
    seed_everything(kwargs.get("seed"), workers=True)

    # Transforms
    image_size = kwargs.get("image_size")
    transforms = T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )

    # Dataset
    d_type = kwargs.get("dataset")
    dataset = get_dataset(d_type, root, transform=transforms)
    N = len(dataset)
    batch_size = kwargs.get("batch_size")
    batch_size = min(N, batch_size)

    # Model
    lr = kwargs.get("lr")
    vae = VAE(
        kwargs.get("enc_block_config"),
        kwargs.get("dec_block_config"),
        kwargs.get("enc_channel_config"),
        kwargs.get("dec_channel_config"),
        lr=lr,
        alpha=1.0,
    )

    # Trainer
    train_kwargs = {}
    restore_path = kwargs.get("restore_path")
    if restore_path is not None:
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    results_dir = kwargs.get("results_dir")
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename="vae-{epoch:02d}-{train_loss:.2f}",
        every_n_epochs=kwargs.get("chkpt_interval", 1),
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = kwargs.get("epochs")
    train_kwargs["log_every_n_steps"] = kwargs.get("log_step")
    train_kwargs["callbacks"] = [chkpt_callback]

    device = kwargs.get("device")
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
    if kwargs.get("fp16"):
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=kwargs.get("workers"),
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    # Logger
    wandb_logger = WandbLogger(
        name=kwargs.get("wandb_run_name"),
        log_model=False,
        project="vaedm",
    )
    # train_kwargs["logger"] = wandb_logger

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(vae, train_dataloader=loader)


if __name__ == "__main__":
    train()
