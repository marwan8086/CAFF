import torch
import torch.nn as nn


class DBM(nn.Module):
    def __init__(self, d_model: int, rho: int):
        super().__init__()
        self.P = nn.Linear(d_model, d_model)
        self.b_g = nn.Parameter(torch.zeros(d_model))
        self.U = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.V = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

    def forward(self, z_prev: torch.Tensor) -> torch.Tensor:
        z_avg = z_prev.mean(0)  # [d_model]
        gate = torch.sigmoid(self.P(z_avg) + self.b_g)
        modulation = gate.unsqueeze(1) * self.U + (1 - gate).unsqueeze(1) * self.V
        return modulation
