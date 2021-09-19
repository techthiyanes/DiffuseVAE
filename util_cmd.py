# A collection of utility commands
import os
import click
from PIL import Image
from tqdm import tqdm


@click.command()
@click.argument("src-folder")
@click.argument("dst-folder")
@click.argument("size", type=int)
@click.option("--format", default="png")
def resize_src_images(src_folder, dst_folder, size, format="png"):
    os.makedirs(dst_folder, exist_ok=True)
    for i in tqdm(os.listdir(src_folder)):
        img_path = os.path.join(src_folder, i)
        img = Image.open(img_path).resize((size, size))
        img_dst_path = os.path.join(dst_folder, i)

        # Write to dst (in format)
        path, ext = os.path.splitext(img_dst_path)
        img_dst_path = path + f".{format}"
        img.save(img_dst_path, format=format)


if __name__ == "__main__":
    resize_src_images()
