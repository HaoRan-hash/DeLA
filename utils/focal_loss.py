import torch
from torch import nn
import torch.nn.functional as F


class PolyFocalLoss(nn.Module):
    def __init__(self, gamma, alpha, epsilon):
        super(PolyFocalLoss, self). __init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, y_pred, y):
        """
        y_pred.shape = (b*n, c)
        y.shape = (b*n, )
        """
        p = torch.sigmoid(y_pred)
        
        y = F.one_hot(y, y_pred.shape[1])   # y.shape = (b*n, c)
        y = y.to(dtype=y_pred.dtype)
        
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction='none')
        
        pt = y * p + (1 - y) * (1 - p)
        focal_loss = bce_loss * ((1 - pt) ** self.gamma)
        
        alpha_t = y * self.alpha + (1 - y) * (1 - self.alpha)
        focal_loss = alpha_t * focal_loss
        
        poly_focal_loss = focal_loss + self.epsilon * torch.pow(1 - pt, self.gamma + 1)
        
        return poly_focal_loss.mean()
