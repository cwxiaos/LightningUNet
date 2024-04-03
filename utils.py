import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        # print(inputs.shape, target.shape)
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

import torch
from einops import rearrange, einsum

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, result, gt):
        result = rearrange(result, 'b c h w -> b c (h w)')
        result = torch.softmax(result, dim=1)
        gt = rearrange(gt, 'b h w -> b 1 (h w)')

        y_onehot = torch.zeros_like(result)
        y_onehot = y_onehot.scatter_(1, gt.to(torch.int64), 1)

        intersection = einsum(result, y_onehot, "b c n, b c n -> b c")
        FP = einsum(result, 1-y_onehot, "b c n, b c n -> b c")
        FN = einsum(1-result, y_onehot, "b c n, b c n -> b c")
        denominator = intersection + self.alpha * FP + self.beta * FN
        divided = 1 - einsum(intersection, "b c -> b") / einsum(denominator, "b c -> b").clamp(min=self.smooth)

        return divided.mean()
