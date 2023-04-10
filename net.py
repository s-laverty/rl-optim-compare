'''
This file defines the deep Q artificial neural network.

Created on 3/17/2023 by Steven Laverty (lavers@rpi.edu).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        *,
        scaler: nn.Module | None = None,
        num_layers: int=1,
        hidden_dim: int=128,
    ) -> None:
        super().__init__()
        self.scaler = scaler

        self.encoder = nn.Linear(state_dim, hidden_dim)
        self.hidden_stack = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.decoder = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            x = self.scaler(x)

        x = F.relu(self.encoder(x))

        for hidden in self.hidden_stack:
            x = F.relu(hidden(x))
        
        q = self.decoder(x)

        return q
