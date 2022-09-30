import datetime
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd
import tensorflow as tf
from keras.callbacks import History


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class Config:
    SEED: int = 42
    TRAIN_IMAGE_SIZE: int = 1024
    INFER_IMAGE_SIZE: int = 1632
    EPOCHS: int = 300
    TRAIN_FOLDER: str = f"datasets/0-last-clean-2/train"
    VALID_FOLDER: str = f"datasets/0-last-clean-2/valid"
    INFER_FOLDER: str = f"datasets/infer"
    BACKBONE: str = "resnet34"
    N_CLASSES: int = 2
    ACTIVATION: str = "softmax"
    NUM_WORKERS: int = tf.data.AUTOTUNE
    PREFETCH_BUFFER: int = tf.data.AUTOTUNE
    TRAIN_BATCH_SIZE: int = 2
    INFER_BATCH_SIZE: int = 1
    LR: float = 1e-4
    MAX_LR: float = 2e-3
    REDUCTION: float = 2
    FACTOR: int = 5
    MONITOR_METRIC: str = "val_f1-score"
    METRIC_MODE: str = "max"
    EXPERIMENTS_FOLDER: str = "experiments/"
    MODEL_TYPE: str = "unet"
    EXPERIMENT_NAME: Optional[str] = None
    WEIGHTS: Optional[str] = None
    LOGS_CSV: Optional[str] = None
    LOGS_TB: Optional[str] = None
    SUBMIT: Optional[str] = None
    _timestamp: Optional[str] = None
    restored: bool = False

    def __post_init__(self):
        if not self.restored:
            self._timestamp = get_timestamp()

        self.EXPERIMENT_NAME = f"{self._timestamp}-{self.BACKBONE}-{self.TRAIN_IMAGE_SIZE}-{self.TRAIN_BATCH_SIZE}-{self.LR}-{self.EPOCHS}-{self.TRAIN_FOLDER.split('/')[-2]}-{self.VALID_FOLDER.split('/')[-2]}"
        if not self.restored:
            os.makedirs(self.FOLDER)
        self.WEIGHTS = f"{self.FOLDER}/weights.hdf5"
        self.LOGS_CSV = f"{self.FOLDER}/metrics.csv"
        self.LOGS_TB = f"{self.FOLDER}/tensorboard"
        self.config_path = f"{self.FOLDER}/config.json"

    @property
    def final_submit_folder(self):
        return f"{self.FOLDER}/{self._timestamp}-submit-final"

    @property
    def temp_submit_folder(self):
        return f"{self.FOLDER}/{get_timestamp()}-submit-temp"

    @property
    def FOLDER(self):
        return f"{self.EXPERIMENTS_FOLDER}/{self.MODEL_TYPE}/{self.EXPERIMENT_NAME}"

    def dump(self):
        with open(self.config_path, "w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=4)

    def save_history(self, history: History):
        df = pd.DataFrame(history.history)
        with open(f"{self.FOLDER}/history.json", mode="w") as f:
            df.to_json(f, indent=4)

        with open(f"{self.FOLDER}/history.csv", mode="w") as f:
            df.to_csv(f, sep="\t")

    @classmethod
    def fromfile(cls, filename: str):
        with open(filename, "r", encoding="utf-8") as file:
            _dict = json.load(file)

        if "restored" in _dict.keys():
            _dict.pop("restored")

        return cls(restored=True, **_dict)


# config = Config()
