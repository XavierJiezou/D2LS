from torch import nn
from torch.nn.modules.loss import _Loss

__all__ = ["JointLoss", "WeightedLoss"]


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight




class JointLoss(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)


# class WeightedLoss(_Loss):
#     """
#     Wrapper class around loss function that applies weighted with fixed factor and distance weights.
#     This class helps to balance multiple losses if they have different scales and incorporate distance weights.
#     """
#
#     def __init__(self, loss, weight=1.0, distance_weights=None):
#         super().__init__()
#         self.loss = loss
#         self.weight = weight
#         self.distance_weights = distance_weights
#
#     def forward(self, logits, labels, distances):
#         loss_value = self.loss(logits, labels, distances)
#
#         if self.distance_weights is not None:
#             weighted_distances = distances * self.distance_weights
#             loss_value = loss_value + (weighted_distances * self.weight)
#         else:
#             loss_value = loss_value * self.weight
#
#         return loss_value
#
# class JointLoss(_Loss):
#     """
#     Wrap two loss functions into one. This class computes a weighted sum of two losses
#     and incorporates distance information.
#     """
#
#     def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0, distance_weights=None):
#         super().__init__()
#         self.first = WeightedLoss(first, first_weight, distance_weights)
#         self.second = WeightedLoss(second, second_weight, distance_weights)
#
#     def forward(self, logits, labels, distances):
#         return self.first(logits, labels, distances) + self.second(logits, labels, distances)
