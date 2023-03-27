'''
This file defines a wrapper for converting observations to pytorch
tensors.

Created on 3/27/2023 by Steven Laverty (lavers@rpi.edu).
'''

import gymnasium as gym
import torch


class TorchWrapper(gym.ObservationWrapper):
    def observation(self, observation) -> torch.Tensor:
        return torch.as_tensor(observation, torch.float)
