"""Faster R-CNN model factory."""
from __future__ import annotations
import logging
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

LOGGER = logging.getLogger(__name__)

def build_model(num_classes: int, pretrained: bool = True):
    """Build Faster R-CNN with a class-count-specific prediction head."""
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        LOGGER.warning('Could not load pretrained weights (%s); using random initialization.', exc)
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model
