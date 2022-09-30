from typing import List

import numpy as np
import segmentation_models as sm
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from keras.losses import Loss
from keras.optimizers import Optimizer

from config import Config
from source.augmentations import Stages, get_transforms
from source.dataset import create_tvi_data, get_data_generator
from source.model import get_model
from source.utils import seed_everything


def get_tf_dataset(
    data, preprocessing, input_size, batch_size, prefetch_buffer, num_workers
) -> tf.data.Dataset:
    return (
        tf.data.Dataset.from_generator(
            lambda: get_data_generator(data, preprocessing=preprocessing),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                [input_size, input_size, 3],
                [input_size, input_size, 2],
            ),
        )
        .map(lambda x, y: (x, y), num_parallel_calls=num_workers)
        .prefetch(buffer_size=prefetch_buffer)
        .batch(batch_size)
    )


def get_optimizer(
    steps_per_epoch, init_lr, max_lr, reduction=1.2, factor=8
) -> Optimizer:
    clr = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=init_lr,
        maximal_learning_rate=max_lr,
        scale_fn=lambda x: 1 / (reduction ** (x - 1)),
        step_size=factor * steps_per_epoch,
    )
    return tf.keras.optimizers.Adam(clr)


def get_callbacks(path_to_weights, path_to_csv, path_to_tb) -> List[Callback]:
    return [
        EarlyStopping(patience=200, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            path_to_weights,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        CSVLogger(
            path_to_csv,
            separator="\t",
        ),
        TensorBoard(path_to_tb),
    ]


def get_loss() -> Loss:
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.8, 0.2]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    return total_loss


def get_metrics():
    return [
        sm.metrics.FScore(threshold=0.5),
    ]


def write_completion_status(folder: str, status: str):
    with open(f"{folder}/complete.txt", "w") as file:
        file.write(status)


if __name__ == "__main__":
    C = Config()
    sm.set_framework("tf.keras")
    seed_everything(C.SEED)

    preprocess_input, model = get_model(
        C.MODEL_TYPE, C.BACKBONE, C.N_CLASSES, C.ACTIVATION
    )
    model.summary()

    total_loss = get_loss()
    metrics = get_metrics()

    train_transform = get_transforms(C.TRAIN_IMAGE_SIZE, Stages.TRAIN)
    valid_transform = get_transforms(C.INFER_IMAGE_SIZE, Stages.VALID)

    train_data, valid_data, infer_data = create_tvi_data(
        train_folder=C.TRAIN_FOLDER,
        valid_folder=C.VALID_FOLDER,
        infer_folder=C.INFER_FOLDER,
        train_transform=train_transform,
        valid_transform=valid_transform,
        infer_transform=valid_transform,
    )

    train_dataset = get_tf_dataset(
        train_data,
        preprocess_input,
        C.TRAIN_IMAGE_SIZE,
        C.TRAIN_BATCH_SIZE,
        C.PREFETCH_BUFFER,
        C.NUM_WORKERS,
    )
    valid_dataset = get_tf_dataset(
        valid_data,
        preprocess_input,
        C.INFER_IMAGE_SIZE,
        C.INFER_BATCH_SIZE,
        C.PREFETCH_BUFFER,
        C.NUM_WORKERS,
    )

    steps_per_epoch = len(train_data) // C.TRAIN_BATCH_SIZE
    optimizer = get_optimizer(
        steps_per_epoch, C.LR, C.MAX_LR, reduction=C.REDUCTION, factor=C.FACTOR
    )

    callbacks = get_callbacks(C.WEIGHTS, C.LOGS_CSV, C.LOGS_TB)
    C.dump()
    try:
        model.compile(optimizer, total_loss, metrics=metrics)
        history = model.fit(
            train_dataset,
            epochs=C.EPOCHS,
            verbose=1,
            validation_data=valid_dataset,
            callbacks=callbacks,
        )
        if history is not None:
            C.save_history(history)
        write_completion_status(C.FOLDER, "complete")
    except KeyboardInterrupt:
        write_completion_status(C.FOLDER, "interupted")
    except Exception as ex:
        write_completion_status(C.FOLDER, str(ex))
        raise ex
