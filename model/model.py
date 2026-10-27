# -*- coding: utf-8 -*-
"""
DualBranchNet — dual-branch spatio-temporal EEG model:
input   (B, C, T)            raw EEG epoch, C electrodes x T samples
output  (B, n_classes)       logits (softmax is applied outside)
"""

from typing import Optional

import torch
import torch.nn as nn

from dcnv4 import DCNv4


class Embedding(nn.Module):
    def __init__(self, in_ch: int, d: int = 5, ksize: int = 3, stride: int = 2):
        super().__init__()
        if ksize % 2 == 0:
            raise ValueError("kernel size must be odd")
        self.d = d
        self.ksize = ksize
        self.stride = stride
        self.padding = (ksize - 1) // 2
        self.conv = nn.Conv1d(
            in_ch, in_ch * d, ksize, stride=stride, padding=self.padding, groups=in_ch
        )
        self.proj = nn.Linear(d, d)

    def emit_length(self, T: int) -> int:
        return (T + 2 * self.padding - self.ksize) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, ch, _ = x.shape
        x = self.conv(x)                              # (B, C*d, L)
        L = x.shape[-1]
        x = x.reshape(bs, ch, self.d, L)              # (B, C, d, L)
        x = self.proj(x.permute(0, 1, 3, 2))          # (B, C, L, d)
        return x.permute(0, 1, 3, 2).contiguous()     # (B, C, d, L)


class SE(nn.Module):
    def __init__(self, token_dim: int, reduction: int = 4, token_axis: int = 1):
        super().__init__()
        if token_axis not in (1, -1):
            raise ValueError("token_axis must be 1 or -1")
        self.token_axis = token_axis
        hidden = max(token_dim // reduction, 1)
        self.excitation = nn.Sequential(
            nn.Linear(token_dim, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, token_dim, bias=False),
        )
        self.gate = nn.Sigmoid()

    @property
    def feature_axis(self) -> int:
        return -1 if self.token_axis == 1 else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(self.excitation(x.mean(dim=self.feature_axis)))
        return w.unsqueeze(self.feature_axis) * x


class PWConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1),
            nn.GELU(),
            nn.Conv1d(out_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DWConv(nn.Module):
    def __init__(self, in_dim: int, ksize: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, ksize, padding=padding, groups=in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ksize: int = 3,
                 se_reduction: int = 4, token_axis: int = 1, token_dim: int = 0):
        super().__init__()
        self.dwconv = DWConv(in_dim, ksize=ksize)
        self.pwconv = PWConv(in_dim, out_dim)
        self.se = SE(token_dim, reduction=se_reduction, token_axis=token_axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.dwconv(x)
        net = self.pwconv(net) + net
        return self.se(net)


class MeanPool(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class Transpose(nn.Module):
    def __init__(self, dim0: int = 1, dim1: int = 2):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class TA(nn.Module):
    def __init__(self, in_dim: int, red: Optional[int] = None):
        super().__init__()
        red = red if red else max(in_dim // 4, 1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, red, 1),
            nn.BatchNorm1d(red),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(red, red, 3, padding=1),
            nn.BatchNorm1d(red),
            nn.ELU(),
        )
        self.att = nn.Sequential(
            nn.Conv1d(red, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.conv2(self.conv1(x))               # (B, 2h/4, N)
        return self.att(net).squeeze(1)               # (B, N)


class LSTMBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, num_layers: int = 1,
                 ta_red: Optional[int] = None):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden, batch_first=True, bidirectional=True, num_layers=num_layers
        )
        self.ta = TA(hidden * 2, red=ta_red)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.lstm(x)[0]                         # (B, N, 2h)
        w = self.ta(net.transpose(1, 2))              # (B, 2h, N) -> (B, N)
        net = net * w.unsqueeze(-1)
        return net.mean(dim=1)                        # (B, 2h)


class TransformerEncoder(nn.Module):
    def __init__(self, n_tokens: int, expansion: int = 4, dropout: float = 0.0,
                 attn_scale: str = "token"):
        super().__init__()
        if attn_scale not in ("token", "n_tokens"):
            raise ValueError("attn_scale must be 'token' or 'n_tokens'")
        self.attn_scale = attn_scale
        self.n_tokens = n_tokens
        self.qkv = nn.Linear(n_tokens, n_tokens * 3)
        self.ln_1 = nn.LayerNorm(n_tokens)
        self.ln_2 = nn.LayerNorm(n_tokens)
        self.dropout = nn.Dropout(dropout)
        self.pffn = nn.Sequential(
            nn.Linear(n_tokens, n_tokens * expansion),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Linear(n_tokens * expansion, n_tokens),
        )

    def _scale(self, token_dim: int) -> float:
        if self.attn_scale == "n_tokens":
            return self.n_tokens ** -0.5
        return token_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, W) — electrodes are tokens, W is the token width
        x_t = x.transpose(1, 2)                       # (B, W, C)
        qkv = self.qkv(x_t).transpose(1, 2)           # (B, 3C, W)
        q, k, v = qkv.chunk(3, dim=1)

        att = torch.softmax(q @ k.transpose(-1, -2) * self._scale(x.shape[-1]), dim=-1)
        att = self.dropout(att @ v)

        net = self.ln_1((att + x).transpose(1, 2)).transpose(1, 2)
        pffn_out = self.pffn(net.transpose(1, 2)).transpose(1, 2)
        return self.ln_2((pffn_out + net).transpose(1, 2)).transpose(1, 2)


class SpatialDCN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dcn = DCNv4(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dcn(x.unsqueeze(1)).squeeze(1)     # (B, C, W) -> (B, 1, C, W) -> (B, C, W)


class Transformer(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int, out_dim: int = 64, depth: int = 1,
                 dropout: float = 0.0, use_cuda_dcn: bool = True,
                 attn_scale: str = "token", align_hidden: Optional[int] = None):
        super().__init__()
        self.dcn = SpatialDCN(
            channels=1, kernel_size=(3, 1), stride=1, pad=(1, 0),
            group=1, use_cuda=use_cuda_dcn,
        )
        self.blocks = nn.Sequential(
            *[TransformerEncoder(in_dim, dropout=dropout, attn_scale=attn_scale)
              for _ in range(depth)]
        )
        self.pool = MeanPool(dim=1)
        self.align = (
            nn.Sequential(nn.Linear(feat_dim, align_hidden), nn.ELU(),
                          nn.Linear(align_hidden, out_dim))
            if align_hidden else nn.Linear(feat_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.blocks(self.dcn(x))                # (B, C, W)
        return self.align(self.pool(net))             # (B, W) -> (B, out_dim)


# full model
class DualBranchNet(nn.Module):
    """Dual-branch spatio-temporal network: temporal (TCN + LSTM) + spatial (TCN + attention)"""

    def __init__(
        self,
        in_dim: int = 63,
        d: int = 5,
        times: int = 500,
        o_dim: int = 64,
        n_classes: int = 2,
        use_cuda_dcn: bool = True,
        lstm_layers: int = 1,
        lstm_hidden: int = 32,
        ksize: int = 3,
        stride: int = 2,
        head_hidden: int = 64,
        dropout: float = 0.0,
        attn_scale: str = "token",
        align_hidden: Optional[int] = None,
        temp_tcn: bool = True,
        lstm: bool = True,
        spat_tcn: bool = True,
        transformer: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d = d
        self.times = times

        self.embedding = Embedding(in_dim, d=d, ksize=ksize, stride=stride)
        self.L = self.embedding.emit_length(times)

        # temporal branch: (B, C*d, L) -> (B, L, C*d)
        self.temporal_tcn = nn.Sequential(
            Transpose(1, 2),
            *([TCN(self.L, self.L, token_axis=1, token_dim=self.L)] if temp_tcn else []),
        )
        self.temporal_lstm = (
            LSTMBlock(d * in_dim, hidden=lstm_hidden, num_layers=lstm_layers)
            if lstm else nn.Sequential(MeanPool(-1), nn.Linear(self.L, o_dim))
        )

        # spatial branch: (B, C, d*L)
        self.spatial_tcn = (
            TCN(in_dim, in_dim, token_axis=1, token_dim=in_dim)
            if spat_tcn else nn.Identity()
        )
        self.transformer = (
            Transformer(
                in_dim=in_dim, feat_dim=d * self.L, out_dim=o_dim,
                dropout=dropout, use_cuda_dcn=use_cuda_dcn,
                attn_scale=attn_scale, align_hidden=align_hidden,
            )
            if transformer else nn.Sequential(MeanPool(-1), nn.Linear(in_dim, o_dim))
        )

        # classification head
        temporal_width = lstm_hidden * 2 if lstm else o_dim
        self.fc = nn.Sequential(
            nn.Linear(o_dim + temporal_width, head_hidden),
            *([nn.Dropout(dropout)] if dropout > 0 else []),
            nn.ELU(),
            nn.Linear(head_hidden, head_hidden // 2),
            *([nn.Dropout(dropout)] if dropout > 0 else []),
            nn.ELU(),
            nn.Linear(head_hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedd = self.embedding(x)                    # (B, C, d, L)
        bs, ch, d, l = embedd.shape

        temporal = self.temporal_lstm(self.temporal_tcn(embedd.reshape(bs, ch * d, l)))
        spatial = self.transformer(self.spatial_tcn(embedd.reshape(bs, ch, d * l)))

        return self.fc(torch.cat([spatial, temporal], dim=1))
