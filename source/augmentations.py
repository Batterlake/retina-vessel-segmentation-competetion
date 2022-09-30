from enum import Enum, auto

import albumentations as A


class Stages(Enum):
    TRAIN = auto()
    VALID = auto()
    INFER = auto()


def get_transforms(size: int, stage: Stages) -> A.Compose:
    transforms_list = [
        A.CLAHE(always_apply=True),
    ]

    if stage is Stages.TRAIN:
        transforms_list += [
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.Transpose(p=0.5),
            # A.RandomRotate90(p=0.5),
            # A.ShiftScaleRotate(
            #     shift_limit=0.2, scale_limit=0.3, rotate_limit=45, p=0.5
            # ),
            A.OneOf(
                [
                    A.Compose(
                        [
                            A.PadIfNeeded(size, size, always_apply=True),
                            A.CropNonEmptyMaskIfExists(size, size, always_apply=True),
                        ]
                    ),
                    A.Compose(
                        [
                            A.Resize(size, size, always_apply=True),
                        ]
                    ),
                    A.Compose(
                        [
                            A.LongestMaxSize(size, always_apply=True),
                            A.PadIfNeeded(size, size, always_apply=True),
                            A.CropNonEmptyMaskIfExists(size, size, always_apply=True),
                        ]
                    ),
                    A.Compose(
                        [
                            A.SmallestMaxSize(size, always_apply=True),
                            A.PadIfNeeded(size, size, always_apply=True),
                            A.CropNonEmptyMaskIfExists(size, size, always_apply=True),
                        ]
                    ),
                ],
                p=1,
            )
        ]
    else:
        transforms_list += [
            A.LongestMaxSize(size, always_apply=True),
            A.PadIfNeeded(size, size, always_apply=True),
        ]
    return A.Compose(transforms_list)
