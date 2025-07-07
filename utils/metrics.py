import torch


def pixel_accuracy(preds, masks):
    """Calculate pixel accuracy (percentage of correct pixels)"""
    correct = (preds == masks).float().sum()
    total = masks.numel()
    return (correct / total).item()


def compute_iou(preds, masks, eps=1e-7):
    """Intersection over Union (IoU)"""
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    return ((intersection + eps) / (union + eps)).item()
