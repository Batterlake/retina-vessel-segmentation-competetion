from enum import Enum, auto

import albumentations as A
import cv2


class Stages(Enum):
    TRAIN = auto()
    VALID = auto()
    INFER = auto()


def get_transforms(size: int, stage: Stages) -> A.Compose:
    transforms_list = [
        # A.CLAHE(always_apply=True),
    ]

    if stage is Stages.TRAIN:
        transforms_list += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.OneOf(
            #     [
            #         A.RandomContrast(),
            #         A.RandomGamma(),
            #         A.RandomBrightness(),
            #     ],
            #     p=0.3,
            # ),
            # A.OneOf(
            #     [
            #         A.ElasticTransform(
            #             alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
            #         ),
            #         A.GridDistortion(),
            #         A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            #     ],
            #     p=0.3,
            # ),
            # A.OneOf(
            #     [
            #         A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.1 * 255)),
            #         A.GaussNoise(var_limit=(10, 80)),
            #     ],
            #     p=0,
            # ),
            # A.OneOf(
            #     [
            #         A.MotionBlur(blur_limit=11, p=cv2.INTER_CUBIC),
            #         A.MedianBlur(blur_limit=11, p=cv2.INTER_CUBIC),
            #         A.Blur(blur_limit=5, p=1),
            #     ],
            #     p=0.4,
            # ),
            # A.ShiftScaleRotate(
            #     shift_limit=0.1, scale_limit=(-0.4, 0.2), rotate_limit=45, p=0.5
            # ),
            # A.OneOf(
            #     [A.HueSaturationValue(p=1), A.ColorJitter(p=1), A.ChannelShuffle(p=1)],
            #     p=1,
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
            ),
        ]
    else:
        transforms_list += [
            A.CLAHE(always_apply=True),
            A.LongestMaxSize(size, always_apply=True),
            A.PadIfNeeded(size, size, always_apply=True),
        ]
    return A.Compose(transforms_list)
