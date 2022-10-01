import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import albumentations as A
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tqdm import tqdm, trange

from config import Config
from source.augmentations import Stages, get_transforms
from source.model import get_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", dest="config_file", help="path to config file of trained_model", type=str
    )
    parser.add_argument(
        "-df", dest="data_folder", help="path to folder with data to infer", type=str
    )

    return parser.parse_args()


def predict(image, model, preprocess):
    patches_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    if len(patches_tensor.shape) == 3:
        patches_tensor = tf.expand_dims(patches_tensor, axis=0)
    patches_tensor = preprocess(patches_tensor)
    preds = model(patches_tensor)
    return preds


if __name__ == "__main__":
    args = parse_args()
    C = Config.fromfile(args.config_file)
    preprocessing, model = get_model(
        C.MODEL_TYPE, C.BACKBONE, C.N_CLASSES, C.ACTIVATION
    )
    model.load_weights(C.WEIGHTS)

    image_files = [x.name for x in Path(args.data_folder).glob("*.png")]
    images = [
        np.asarray(Image.open(f"{args.data_folder}/{file}")) for file in image_files
    ]
    transform = get_transforms(C.INFER_IMAGE_SIZE, Stages.INFER)
    orig_size = (1232, 1624)
    inverse_transform = A.Compose(
        [
            A.CenterCrop(*orig_size),
        ]
    )

    transformed_images = [transform(image=image)["image"] for image in images]

    folder = C.temp_submit_folder
    os.makedirs(folder, exist_ok=True)

    for index in trange(0, len(transformed_images)):
        psum = predict(transformed_images[index], model, preprocessing)
        mask_pred = tf.argmax(psum, axis=-1).numpy().astype(np.uint8)[0] * 255
        mask_pred = inverse_transform(image=mask_pred)["image"]
        Image.fromarray(mask_pred).save(f"{folder}/{image_files[index]}")

    p = Path(folder)
    shutil.make_archive(str(p.parent / os.path.splitext(p.name)[0]), "zip", folder)
