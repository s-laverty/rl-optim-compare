'''
This file defines a wrapper for converting observations to pytorch
tensors.

Created on 3/27/2023 by Steven Laverty (lavers@rpi.edu).
'''

import gymnasium as gym
import torch


class TorchWrapper(gym.ObservationWrapper):
    dtype: torch.dtype
    device: torch.device
    batch_dim: bool

    def __init__(
        self,
        env: gym.Env,
        dtype: torch.dtype = torch.float,
        device: torch.device = 'cpu',
        batch_dim: bool = False,
    ) -> None:
        super().__init__(env)
        self.dtype = dtype
        self.device = device
        self.batch_dim = batch_dim

    def observation(self, observation) -> torch.Tensor:
        tensor = torch.as_tensor(observation, self.dtype, self.device)
        if self.batch_dim:
            tensor = tensor.unsqueeze(0)
        return tensor
