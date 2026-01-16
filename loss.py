import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, probs.shape[1]).permute(0,3,1,2)

        inter = (probs * targets_oh).sum((2,3))
        union = probs.sum((2,3)) + targets_oh.sum((2,3))

        dice = (2*inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
