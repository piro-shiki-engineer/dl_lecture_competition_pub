import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size: int = 3, p_drop: float = 0.1) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.projection = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        
        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        
        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        identity = self.projection(X)
        
        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))
        X = self.dropout(X)
        
        X = self.conv1(X)
        X = F.gelu(self.batchnorm1(X))
        
        X = X + identity
        return self.dropout(X)

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        return self.head(X)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x.permute(0, 2, 1)  # (batch, channels, seq_len)

class ImprovedConvClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 256) -> None:
        super().__init__()
        
        self.input_proj = nn.Conv1d(in_channels, hid_dim, kernel_size=1)
        
        self.blocks = nn.ModuleList([
            ConvBlock(hid_dim, hid_dim, p_drop=0.1) for _ in range(6)
        ])
        
        self.attention_blocks = nn.ModuleList([
            SelfAttention(hid_dim) for _ in range(2)
        ])
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Dropout(0.2),
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid_dim // 2, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.input_proj(X)
        
        for i, block in enumerate(self.blocks):
            X = block(X)
            if i % 3 == 2:  # Every 3rd block, apply attention
                X = self.attention_blocks[i // 3](X)
        
        return self.head(X)