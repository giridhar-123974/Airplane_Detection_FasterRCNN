"""Image transforms for object detection."""
from __future__ import annotations

import random
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, target: dict):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(__import__('numpy').asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        return image, target


class RandomHorizontalFlip:
    def __init__(self, probability: float = 0.5):
        self.probability = probability

    def __call__(self, image: torch.Tensor, target: dict):
        if random.random() < self.probability:
            image = image.flip(-1)
            width = image.shape[-1]
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target = {**target, 'boxes': boxes}
        return image, target


def get_transforms(train: bool):
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip())
    return Compose(transforms)
