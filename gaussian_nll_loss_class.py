import numpy as np
import math
import torch


class GaussianNLLLoss(torch.nn.Module):
    def __init__(self, eps=1e-8, full=False, reduction='mean'):
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.full = full
        self.reduction = reduction

    def forward(self, input, target, var):
        #Predictions and targets much have same shape
        if input.shape != target.shape:
            raise ValueError("input and target must have same shape!")

        #first dimension N must be equal
        if var.shape[0] != input.shape[0]:
            raise ValueError(f"dim 0 of var and input do not match!")

        #second dim must be 1 ie shape (N, 1) if var and input do not have same shape
        if var.shape != input.shape and len(var.shape) > 1 and var.shape[1] != 1:
            raise ValueError("var is of incorrect shape!")

        #Add eps for stability
        var = var + self.eps

        #Calculate loss (without constant)
        loss = 0.5*(torch.sum(torch.log(var) + (input - target)**2/var, dim=1))

        #Add constant to loss term if required
        if self.full:
            length = input.shape[1]
            loss = loss + 0.5*length*math.log(2*math.pi)

        #Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
