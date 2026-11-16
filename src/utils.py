"""Shared utilities for training and model persistence."""
from __future__ import annotations
import logging, random
from pathlib import Path
import numpy as np
import torch

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s | %(levelname)s | %(message)s')

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(path, model, optimizer, scheduler, epoch, class_names, history):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'class_names': class_names, 'history': history}, path)
