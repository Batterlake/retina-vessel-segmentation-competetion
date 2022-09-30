import matplotlib.pylab as plt
import numpy as np
from PIL import Image


def t2i(tensor):
    return Image.fromarray(tensor.numpy().astype(np.uint8))


def tensor_to_image(tensor):
    return (tensor["image"]).astype(np.uint8)


def tensor_to_mask(tensor):
    return 255 - np.clip((tensor["mask"] * 255)[:, :], 0, 255).astype(np.uint8)


def overlay_image_and_mask(tensor, color=(255, 0, 0)):
    image = Image.fromarray(tensor_to_image(tensor))
    mask = Image.fromarray(tensor_to_mask(tensor))
    return np.asarray(
        Image.composite(image, Image.new("RGB", image.size, color), mask.convert("L"))
    )
