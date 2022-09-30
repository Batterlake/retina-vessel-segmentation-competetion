from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import tensorflow as tf


class StageDataset:
    def __init__(
        self,
        folder: str,
        filenames: List[str],
        size: Union[Tuple[int, int], int],
        transform: Optional[A.Compose] = None,
        train: bool = True,
    ):
        if isinstance(size, int):
            size = (size, size)

        self.size = size
        self.folder = folder
        self.filenames = filenames
        self.train = train

        self.transform = transform

    @staticmethod
    def decode_img(
        filename: str,
        size: Tuple[int, int],
        channels: int,
        interpolation="bicubic",
        antialias=True,
    ):
        img = tf.io.read_file(filename)
        img = tf.io.decode_image(img, channels=channels)
        if size is not None:
            img = tf.image.resize(img, size, method=interpolation, antialias=antialias)
        img = tf.clip_by_value(img, 0, 255)
        img = tf.cast(img, dtype=tf.uint8)
        return img

    @staticmethod
    def imread(filename: str, size: Tuple[int, int]) -> tf.Tensor:
        return StageDataset.decode_img(filename, size, 3).numpy()

    @staticmethod
    def maskread(filename: str, size: Tuple[int, int]) -> tf.Tensor:
        mask = StageDataset.decode_img(
            filename, size, 1, interpolation="nearest", antialias=False
        )
        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.squeeze(tf.cast(mask, tf.uint8))
        return mask.numpy()

    def __getitem__(self, index) -> Dict[str, tf.Tensor]:
        args1 = (f"{self.folder}/images/{self.filenames[index]}", self.size)
        args2 = (f"{self.folder}/masks/{self.filenames[index]}", self.size)
        output = {"image": StageDataset.imread(*args1)}
        if self.train:
            mask = StageDataset.maskread(*args2)
            output["mask"] = mask

        if self.transform:
            output = self.transform(**output)

        if self.train:
            output["mask_onehot"] = tf.one_hot(output["mask"], depth=2).numpy()

        return output

    def __len__(self):
        return len(self.filenames)


def get_data_generator(stage_data: StageDataset, preprocessing, mask_key="mask_onehot"):
    for item in iter(stage_data):
        image = preprocessing(item["image"])
        mask = item[mask_key]
        yield image, mask


def glob_filenames(folder: str, ext: str):
    return [x.name for x in Path(folder).glob(f"*.{ext}")]


def get_train_data(
    stage_folder: str,
    transform: Optional[A.Compose] = None,
):
    images = set(glob_filenames(f"{stage_folder}/images", "png"))
    masks = set(glob_filenames(f"{stage_folder}/masks", "png"))
    images_w_masks = sorted(
        list(images.intersection(masks)), key=lambda x: int(x.split(".")[0])
    )
    stage_ds = StageDataset(stage_folder, images_w_masks, None, transform=transform)
    return stage_ds


def get_test_data(
    stage_folder: str,
    transform: Optional[A.Compose] = None,
):
    images = set(glob_filenames(f"{stage_folder}/images", "png"))
    images = sorted(list(images), key=lambda x: int(x.split(".")[0]))
    return StageDataset(stage_folder, images, None, transform=transform, train=False)


def get_stage_data(
    stage_folder: str, transform: Optional[A.Compose] = None, train: bool = True
):
    if train:
        stage = get_train_data(stage_folder, transform)
    else:
        stage = get_test_data(stage_folder, transform)
    return stage


def create_tvi_data(
    train_folder: str,
    valid_folder: str,
    infer_folder: str,
    train_transform,
    valid_transform,
    infer_transform,
):
    train_data = get_stage_data(train_folder, train_transform)
    valid_data = get_stage_data(valid_folder, valid_transform)
    infer_data = get_stage_data(infer_folder, infer_transform, train=False)
    return train_data, valid_data, infer_data


def create_tv_datasets(
    train_data,
    valid_data,
    preprocessing,
    train_input_size,
    valid_input_size,
    train_batch_size,
    valid_batch_size,
    n_workers,
    prefetch_buffer_size,
):
    train_dataset = (
        tf.data.Dataset.from_generator(
            lambda: get_data_generator(train_data, preprocessing=preprocessing),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                [train_input_size, train_input_size, 3],
                [train_input_size, train_input_size, 2],
            ),
        )
        .map(lambda x, y: (x, y), num_parallel_calls=n_workers)
        .prefetch(buffer_size=prefetch_buffer_size)
        .batch(train_batch_size)
    )

    valid_dataset = (
        tf.data.Dataset.from_generator(
            lambda: get_data_generator(valid_data, preprocessing=preprocessing),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                [valid_input_size, valid_input_size, 3],
                [valid_input_size, valid_input_size, 2],
            ),
        )
        .map(lambda x, y: (x, y), num_parallel_calls=n_workers)
        .prefetch(buffer_size=prefetch_buffer_size)
        .batch(valid_batch_size)
    )
    return train_dataset, valid_dataset
