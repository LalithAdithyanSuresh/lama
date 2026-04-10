import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class SelectiveScanPurePyTorch(nn.Module):
    """
    A pure PyTorch implementation of the Selective Scan Mechanism (S6).
    This is slower than the CUDA version but runnable without extra libraries.
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        self.activation = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.d_state * 2) + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def selective_scan(self, u, dt, A, B, C, D):
        """
        u: (B, L, D)
        dt: (B, L, D)
        A: (D, N)
        B: (B, L, N)
        C: (B, L, N)
        D: (D)
        """
        B_size, L, D_size = u.shape
        N = A.shape[1]

        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)) # (B, L, D, N)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2) # (B, L, D, N)

        h = torch.zeros(B_size, D_size, N, device=u.device, dtype=u.dtype)
        outputs = []

        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * u[:, i].unsqueeze(-1)
            y = torch.einsum('bdn,bn->bd', h, C[:, i])
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1) + u * D
        return y

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape
        
        # Initial projection
        xz = self.in_proj(x.permute(0, 2, 3, 1)) # (B, H, W, 2*D_inner)
        x, z = xz.chunk(2, dim=-1)
        
        # 2D Conv
        x = x.permute(0, 3, 1, 2) # (B, D_inner, H, W)
        x = self.activation(self.conv2d(x))
        
        # Flatten for SSM
        L = h * w
        u = x.flatten(2).transpose(1, 2) # (B, L, D_inner)
        
        # Projections for S6
        x_dbl = self.x_proj(u) # (B, L, 2*N + 1)
        dt, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)) # (B, L, D_inner)
        
        A = -torch.exp(self.A_log) # (D_inner, N)
        y = self.selective_scan(u, dt, A, B, C, self.D)
        
        # Output
        y = y * self.activation(z.flatten(2).transpose(1, 2))
        y = self.out_proj(y)
        y = y.transpose(1, 2).view(b, -1, h, w)
        
        return y

class VisionMamba2D(nn.Module):
    """
    Bidirectional 2D Vision Mamba Block
    Scans row-wise and column-wise to capture spatial context.
    """
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_h = SelectiveScanPurePyTorch(d_model, d_state=d_state, expand=expand)
        self.mamba_v = SelectiveScanPurePyTorch(d_model, d_state=d_state, expand=expand)

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        
        # Layer Norm (Spatial)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        
        # Horizontal Scan
        out_h = self.mamba_h(x)
        
        # Vertical Scan (Transpose H/W)
        x_v = x.transpose(2, 3)
        out_v = self.mamba_v(x_v)
        out_v = out_v.transpose(2, 3)
        
        return res + 0.5 * (out_h + out_v)
