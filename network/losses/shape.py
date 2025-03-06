from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ShapeAwareLoss"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class ShapeAwareLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(ShapeAwareLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        dice_loss = DiceLoss()
        # print("y_pred size:", y_pred.size())
        num_class = 6  # 예시로 num_class 값을 6으로 가정
        y_true = y_true.unsqueeze(1).expand(-1, num_class, -1, -1)
        # print("y_true size:", y_true.size())
        dice_loss_value = dice_loss(y_pred, y_true)

        shape_loss = self.alpha * (1 - dice_loss_value) * y_pred.mean()
        combined_loss = dice_loss_value + shape_loss
        return combined_loss

