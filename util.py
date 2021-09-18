from datasets.cifar10 import CIFAR10Dataset
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import models.backbone.resnet as backbone_models
from datasets import (
    CelebADataset,
    CelebAMaskHQDataset,
    ReconstructionDataset,
    CIFAR10Dataset,
)

logger = logging.getLogger(__name__)


def configure_device(device):
    if device.startswith("gpu"):
        if not torch.cuda.is_available():
            raise Exception(
                "CUDA support is not available on your platform. Re-run using CPU or TPU mode"
            )
        gpu_id = device.split(":")[-1]
        if gpu_id == "":
            # Use all GPU's
            gpu_id = "-1"
        return f"cuda:{gpu_id}", gpu_id
    elif device == "tpu":
        if _xla_available:
            return xm.xla_device()
        raise Exception("Install PyTorch XLA to use TPUs")
    elif device == "cpu":
        return "cpu"
    else:
        raise NotImplementedError(f"The device type `{device}` is not supported")


# TODO: As more backbones are added register them with the modelstore
def get_resnet_models(backbone_name, pretrained=False, **kwargs):
    """Returns a backbone and its associated inplanes
    Args:
        backbone_name (str): Requested backbone
        pretrained (bool, optional): If the backbone must be pretrained on ImageNet. Defaults to False.
    Raises:
        NotImplementedError: If the backbone_name is not supported
    """
    backbone = None
    supported_backbones = ["resnet18", "resnet34", "resnet50", "resnet101"]
    if backbone_name not in supported_backbones:
        raise NotImplementedError(f"The backbone {backbone_name} is not supported")

    backbone = getattr(backbone_models, backbone_name)(pretrained=pretrained, **kwargs)
    return backbone


def get_dataset(name, root, transform=None, **kwargs):
    if name == "celeba":
        dataset = CelebADataset(root, transform=transform, **kwargs)
    elif name == "celeba-hq":
        dataset = CelebAMaskHQDataset(root, transform=transform, **kwargs)
    elif name == "recons":
        dataset = ReconstructionDataset(root, transform=transform, **kwargs)
    elif name == "cifar10":
        dataset = CIFAR10Dataset(root, transform=transform, **kwargs)
    else:
        raise NotImplementedError(
            f"The dataset {name} does not exist in our datastore."
        )
    return dataset


def normalize(obj):
    B, C, H, W = obj.shape
    for i in range(3):
        channel_val = obj[:, i, :, :].view(B, -1)
        channel_val -= channel_val.min(1, keepdim=True)[0]
        channel_val /= (
            channel_val.max(1, keepdim=True)[0] - channel_val.min(1, keepdim=True)[0]
        )
        channel_val = channel_val.view(B, H, W)
        obj[:, i, :, :] = channel_val
    return obj


# CREDITS: https://github.com/huggingface/pytorch-pretrained-BigGAN/blob/master/pytorch_pretrained_biggan/utils.py
def convert_to_images(obj):
    """Convert an output tensor from BigGAN in a list of images.
    Params:
        obj: tensor or numpy array of shape (batch_size, channels, height, width)
    Output:
        list of Pillow Images of size (height, width)
    """
    obj = normalize(obj)
    obj = obj.detach().numpy()
    obj = obj.transpose((0, 2, 3, 1))
    obj = np.clip(obj * 256, 0, 255)

    img = []
    for _, out in enumerate(obj):
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img.append(Image.fromarray(out_array))
    return img


def save_as_images(obj, file_name="output"):
    """Convert and save an output tensor from BigGAN in a list of saved images.
    Params:
        obj: tensor or numpy array of shape (batch_size, channels, height, width)
        file_name: path and beggingin of filename to save.
            Images will be saved as `file_name_{image_number}.png`
    """
    img = convert_to_images(obj)

    for i, out in enumerate(img):
        current_file_name = file_name + "_%d.png" % i
        logger.info("Saving image to {}".format(current_file_name))
        out.save(current_file_name, "png")


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
