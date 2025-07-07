import torch


def pixel_accuracy(preds, masks):
    """Calculate pixel accuracy (percentage of correct pixels)"""
    preds_bin = (preds > 0.5).float()
    correct = (preds_bin == masks).float().sum()
    total = masks.numel()
    return (correct / total).item()


def compute_iou(preds, masks, eps=1e-7):
    """Intersection over Union (IoU)"""
    preds_bin = (preds > 0.5).float()
    intersection = (preds_bin * masks).sum()
    union = preds_bin.sum() + masks.sum() - intersection
    return ((intersection + eps) / (union + eps)).item()


def pixel_accuracy_evaluate(preds, masks):
    """Calculate pixel accuracy (percentage of correct pixels) for evaluation"""
    correct = (preds == masks).sum()
    total = masks.shape[0] * masks.shape[1]
    return correct / total


def compute_iou_evaluate(preds, masks, eps=1e-7):
    """Intersection over Union (IoU) for evaluation"""
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    return (intersection + eps) / (union + eps)
