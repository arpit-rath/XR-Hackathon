"""PyTorch models for Member 3: frame-importance regression and video-category classification."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class ImportanceTransformer(nn.Module):
    """Per-timestep importance in [0,1] from a sequence of CLIP features (B,T,768)."""

    def __init__(self, d_in=768, d_model=256, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.cfg = dict(d_in=d_in, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout)
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_model), nn.GELU(), nn.Dropout(dropout))
        self.pe = SinusoidalPE(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(),
                                  nn.Linear(d_model // 2, 1))

    def forward(self, x, pad_mask=None):  # pad_mask: True = padding
        h = self.enc(self.pe(self.inp(x)), src_key_padding_mask=pad_mask)
        return torch.sigmoid(self.head(h).squeeze(-1))


def pool_features(feats):
    """Video-level descriptor: [mean, max] of L2-normalised CLIP features -> 1536-d, L2-normalised."""
    if isinstance(feats, torch.Tensor):
        x = feats
    else:
        x = torch.as_tensor(feats, dtype=torch.float32)
    v = torch.cat([x.mean(0), x.max(0).values])
    return F.normalize(v, dim=-1)


class CategoryHead(nn.Module):
    """Linear probe on pooled CLIP features."""

    def __init__(self, d_in=1536, n_classes=14):
        super().__init__()
        self.cfg = dict(d_in=d_in, n_classes=n_classes)
        self.fc = nn.Linear(d_in, n_classes)

    def forward(self, v):
        return self.fc(v)
