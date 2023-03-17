'''
This file defines the deep Q artificial neural network.

Created on 3/17/2023 by Steven Laverty (lavers@rpi.edu).
'''

import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        *,
        scaler: nn.Module | None = None,
        num_layers: int=6,
        hidden_dim: int=32,
    ) -> None:
        super().__init__()
        self.encoder_relu = nn.ReLU()

        self.scaler = scaler
        self.encoder = nn.Linear(state_dim, hidden_dim)
        self.encoder_relu = nn.ReLU()
        self.hidden_stack = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.hidden_stack_relu = nn.ModuleList(
            nn.ReLU() for _ in range(num_layers)
        )
        self.decoder = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None:
            x = self.scaler(x)

        x = self.encoder_relu(self.encoder(x))

        for linear, relu in zip(self.hidden_stack, self.hidden_stack_relu):
            x = relu(linear(x))
        
        q = self.decoder(x)

        return q
