from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
import numpy as np

from .functional import soft_dice_score

__all__ = ["DistanceBasedDiceLoss"]


# class ModifiedDiceLoss(_Loss):
#     def __init__(self, threshold: float, distance_weights=None):
#         super(ModifiedDiceLoss, self).__init__()
#         self.threshold = threshold
#         self.distance_weights = distance_weights
#
#     def forward(self, y_pred: Tensor, y_true: Tensor, distances: Tensor) -> Tensor:
#         assert y_true.size() == y_pred.size()
#         assert y_true.size() == distances.size()
#
#         # 경계 정보 추출 (Thresholding)
#         boundary_mask = (y_pred > self.threshold).float()
#
#         # 거리 정보에 가중치 적용
#         if self.distance_weights is not None:
#             weighted_distances = distances * self.distance_weights
#         else:
#             weighted_distances = distances
#
#         # 거리 기반 수정된 Dice 손실 계산
#         intersection = (boundary_mask * y_true).sum()
#         union = boundary_mask.sum() + y_true.sum()
#
#         weighted_intersection = (weighted_distances * y_true).sum()
#         weighted_union = weighted_distances.sum() + y_true.sum()
#
#         dice = 1.0 - (2.0 * weighted_intersection + 1e-7) / (weighted_union + intersection + 1e-7)
#
#         return dice


BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind not in {"O", "M", "U", "S"}:
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x

    raise ValueError("Unsupported input type" + str(type(x)))

