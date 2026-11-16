"""Dataset loader for Airbus aircraft polygon annotations."""
from __future__ import annotations

import ast
import logging
import random
import warnings
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = False


def inspect_annotations(csv_path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate and summarize this CSV's verified image/polygon/class schema."""
    frame = pd.read_csv(csv_path)
    required = {'image_id', 'geometry', 'class'}
    if not required.issubset(frame.columns):
        raise ValueError(f"Expected annotation columns {sorted(required)}; found {list(frame.columns)}")
    classes = sorted(frame['class'].dropna().astype(str).unique())
    if not classes:
        raise ValueError('No class labels found in annotations.')
    return frame, {name: index + 1 for index, name in enumerate(classes)}


def create_splits(dataset_dir: str | Path, val_fraction: float = 0.2, seed: int = 42, overwrite: bool = False):
    """Create deterministic image-level train/validation CSVs if absent."""
    root = Path(dataset_dir)
    frame, _ = inspect_annotations(root / 'annotations.csv')
    split_dir = root / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    train_file, val_file = split_dir / 'train.csv', split_dir / 'val.csv'
    if train_file.exists() and val_file.exists() and not overwrite:
        return train_file, val_file
    names = sorted(frame['image_id'].dropna().astype(str).unique().tolist())
    if len(names) < 2:
        raise ValueError('At least two annotated images are needed for a train/validation split.')
    random.Random(seed).shuffle(names)
    n_val = max(1, min(len(names) - 1, round(len(names) * val_fraction)))
    val_names = set(names[:n_val])
    frame[frame['image_id'].isin(val_names)].to_csv(val_file, index=False)
    frame[~frame['image_id'].isin(val_names)].to_csv(train_file, index=False)
    LOGGER.info('Created image-level split: %d train and %d validation images.', len(names)-n_val, n_val)
    return train_file, val_file


class AircraftDataset(Dataset):
    """Loads JPEGs and converts closed polygon pixel coordinates to boxes."""
    def __init__(self, dataset_dir: str | Path, annotations: str | Path | pd.DataFrame | None = None,
                 transforms=None, image_dir: str | Path | None = None,
                 class_names: list[str] | None = None):
        self.root = Path(dataset_dir)
        self.image_dir = Path(image_dir) if image_dir else self.root / 'images'
        source = self.root / 'annotations.csv' if annotations is None else annotations
        if isinstance(source, pd.DataFrame):
            self.frame = source.copy()
        else:
            self.frame = pd.read_csv(source)
        required = {'image_id', 'geometry', 'class'}
        if not required.issubset(self.frame.columns):
            raise ValueError(f"Expected annotation columns {sorted(required)}; found {list(self.frame.columns)}")
        duplicate_mask = self.frame.duplicated(['image_id', 'geometry', 'class'], keep='first')
        if duplicate_mask.any():
            warnings.warn(f'Removing {int(duplicate_mask.sum())} duplicate annotation rows.', RuntimeWarning)
            self.frame = self.frame.loc[~duplicate_mask].copy()
        self.classes = class_names or sorted(self.frame['class'].dropna().astype(str).unique())
        self.class_to_id = {name: i + 1 for i, name in enumerate(self.classes)}
        self.transforms = transforms
        candidates = sorted(self.frame['image_id'].dropna().astype(str).unique())
        self.images = []
        self.records = {}
        for name in candidates:
            image_path = self.image_dir / name
            if not image_path.is_file():
                warnings.warn(f'Skipping annotation image with missing file: {image_path}', RuntimeWarning)
                continue
            try:
                with Image.open(image_path) as image_file:
                    image_file.verify()
            except Exception as exc:
                warnings.warn(f'Skipping corrupted image {image_path}: {exc}', RuntimeWarning)
                continue
            self.images.append(name)
            rows = self.frame[self.frame.image_id.astype(str) == name]
            parsed = []
            for _, row in rows.iterrows():
                try:
                    points = ast.literal_eval(str(row['geometry']))
                    xs = [float(point[0]) for point in points]
                    ys = [float(point[1]) for point in points]
                    if len(xs) < 4 or len(xs) != len(ys):
                        raise ValueError('polygon must contain at least four coordinate pairs')
                    label = str(row['class'])
                    if label not in self.class_to_id:
                        raise ValueError(f'unknown class {label!r}')
                    parsed.append(([min(xs), min(ys), max(xs), max(ys)], self.class_to_id[label]))
                except Exception as exc:
                    warnings.warn(f'Skipping invalid annotation for {name}: {exc}', RuntimeWarning)
            self.records[name] = parsed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        path = self.image_dir / name
        try:
            with Image.open(path) as im:
                image = im.convert('RGB')
        except Exception as exc:
            raise RuntimeError(f'Cannot read image {path}: {exc}') from exc
        w, h = image.size
        boxes, labels = [], []
        for box, label in self.records[name]:
            x1, y1, x2, y2 = box
            x1, x2 = max(0., min(x1, w),), max(0., min(x2, w))
            y1, y2 = max(0., min(y1, h)), max(0., min(y2, h))
            if x2 <= x1 or y2 <= y1:
                warnings.warn(f'Skipping invalid box {box} in {name}', RuntimeWarning)
                continue
            boxes.append([x1, y1, x2, y2]); labels.append(label)
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([index], dtype=torch.int64),
            'area': torch.as_tensor([(b[2]-b[0])*(b[3]-b[1]) for b in boxes], dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64),
        }
        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            from .transforms import ToTensor
            image, target = ToTensor()(image, target)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
