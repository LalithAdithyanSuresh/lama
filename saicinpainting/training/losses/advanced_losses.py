import torch
import torch.nn as nn
import torch.nn.functional as F

class HighFrequencyLoss(nn.Module):
    """
    High-Frequency Loss using Laplacian filter to encourage sharp textures.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.register_buffer('kernel', torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]).float().unsqueeze(0).unsqueeze(0))
        self.weight = weight

    def forward(self, pred, target, mask=None):
        # pred, target: (B, 3, H, W)
        # mask: (B, 1, H, W), where 1 is the region to inpaint
        
        b, c, h, w = pred.shape
        kernel = self.kernel.repeat(c, 1, 1, 1)
        
        pred_high = F.conv2d(pred, kernel, padding=1, groups=c)
        target_high = F.conv2d(target, kernel, padding=1, groups=c)
        
        loss = F.l1_loss(pred_high, target_high, reduction='none')
        
        if mask is not None:
            loss = loss * mask
            
        return loss.mean() * self.weight

class BoundaryLoss(nn.Module):
    """
    Boundary Loss to enforce gradient consistency near mask edges.
    Uses Sobel filters.
    """
    def __init__(self, weight=1.0, dilation=5):
        super().__init__()
        self.weight = weight
        self.dilation = dilation
        
        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float().unsqueeze(0).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).float().unsqueeze(0).unsqueeze(0))

    def get_gradients(self, x):
        b, c, h, w = x.shape
        kx = self.sobel_x.repeat(c, 1, 1, 1)
        ky = self.sobel_y.repeat(c, 1, 1, 1)
        
        grad_x = F.conv2d(x, kx, padding=1, groups=c)
        grad_y = F.conv2d(x, ky, padding=1, groups=c)
        
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    def forward(self, pred, target, mask):
        # mask: 1 in missing region
        # Get boundary region by dilating and eroding the mask
        kernel = torch.ones(1, 1, self.dilation, self.dilation).to(mask.device)
        dilated_mask = torch.clamp(F.conv2d(mask, kernel, padding=self.dilation//2), 0, 1)
        eroded_mask = 1 - torch.clamp(F.conv2d(1 - mask, kernel, padding=self.dilation//2), 0, 1)
        
        boundary_mask = dilated_mask - eroded_mask
        boundary_mask = torch.clamp(boundary_mask, 0, 1)
        
        pred_grad = self.get_gradients(pred)
        target_grad = self.get_gradients(target)
        
        loss = F.l1_loss(pred_grad, target_grad, reduction='none')
        loss = loss * boundary_mask
        
        return loss.mean() * self.weight
