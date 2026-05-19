import math

import torch
import torch.nn as nn


class TorchMLP(nn.Module):
    def __init__(self, din, dh, dout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, dh),
            nn.ReLU(),
            nn.Linear(dh, dh),
            nn.ReLU(),
            nn.Linear(dh, dout),
        )

    def forward(self, x):
        return self.net(x)


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class CondDenoiser(nn.Module):
    def __init__(self, zdim, tdim=64, cdim=32, hidden=256):
        super().__init__()
        self.t = TimeEmbed(tdim)
        self.net = nn.Sequential(
            nn.Linear(zdim + tdim + cdim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, zdim),
        )

    def forward(self, zt, t, cond_vec):
        return self.net(torch.cat([zt, self.t(t), cond_vec], dim=-1))
