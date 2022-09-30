from enum import Enum
from typing import Callable, Tuple, Union

import keras as keras
import numpy as np
import segmentation_models as sm

Preprocessing = Callable[[np.ndarray], np.ndarray]


class AvailableModels(Enum):
    UNET = "unet"
    FPN = "fpn"


def _get_model(
    _object, backbone: str, n_classes: int, activation: str
) -> Tuple[Preprocessing, keras.Model]:
    preprocess_input = sm.get_preprocessing(backbone)
    model: keras.Model = _object(
        backbone,
        encoder_weights="imagenet",
        classes=n_classes,
        activation=activation,
    )
    return preprocess_input, model


def get_fpn(
    backbone: str, n_classes: int, activation: str
) -> Tuple[Preprocessing, keras.Model]:
    return _get_model(sm.FPN, backbone, n_classes, activation)


def get_unet(
    backbone: str, n_classes: int, activation: str
) -> Tuple[Preprocessing, keras.Model]:

    return _get_model(sm.Unet, backbone, n_classes, activation)


def get_model(
    _type: Union[str, AvailableModels], backbone: str, n_classes: int, activation: str
) -> Tuple[Preprocessing, keras.Model]:
    if isinstance(_type, str):
        _type = AvailableModels(_type)

    if _type == AvailableModels.FPN:
        func = get_fpn
    else:
        func = get_unet

    return func(backbone, n_classes, activation)
