'''
This file defines common utility functions and data classes
'''

from contextlib import nullcontext
import enum
import json
import os
import pathlib
import typing
from collections import deque

import torch

OBS_SHAPE = 4
NUM_ACTIONS = 2


class OptimType(str, enum.Enum):
    RMS_PROP = 'rms_prop'
    ADAM = 'adam'
    ADAMAX = 'adamax'
    ADAM_W = 'adam_w'
    R_ADAM = 'r_adam'


class OptimConfig(typing.TypedDict, total=False):
    name: str
    optim_type: OptimType
    lr: float
    eps: float
    weight_decay: float
    rms_alpha: float
    rms_momentum: float
    rms_centered: bool
    adam_betas: tuple[float, float]
    adam_amsgrad: bool


class Config(typing.TypedDict):
    name: str
    checkpoint_dir: str
    hidden_layers: int
    hidden_dim: int
    num_models: int
    q_lr: float
    q_gamma: float
    q_epsilon_max: float
    q_epsilon_min: float
    q_epsilon_decay: float
    q_replay_buf_len: int
    q_target_update_tau: float
    train_steps: int
    batch_size: int
    num_eval: int
    checkpoint_steps: int
    optimizers: list[OptimConfig]


class Experience(typing.NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor | None


class Checkpoint(typing.NamedTuple):
    num_steps: int
    policy_net_state_dict: dict
    target_net_state_dict: dict
    optimizer_state_dict: dict
    replay_buffer: deque[Experience]
    eval_hist: list[float]
    best_policy_net_state_dict: dict


def save_checkpoint(
    file: str | bytes | os.PathLike | typing.BinaryIO,
    checkpoint: Checkpoint,
) -> None:
    if isinstance(file, (str, bytes, os.PathLike)):
        dirname = os.path.dirname(file)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
    torch.save(checkpoint, file)


def load_checkpoint(
    file: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    device: torch.device = 'cpu',
) -> Checkpoint:
    return torch.load(file, map_location=device)


def checkpoint_path(config: Config, iteration: int) -> pathlib.Path:
    name = 'checkpoint_{}_iter_{:04d}'.format(config['name'], iteration)
    return pathlib.Path(config['checkpoint_dir']).joinpath(name)


def checkpoint_name(rank: int, optim_rank: int) -> pathlib.Path:
    return pathlib.Path('{:03d}_{:02d}.pt'.format(rank, optim_rank))


def latest_checkpoint_iteration(config: Config) -> int:
    iteration = -1
    glob_pattern = 'checkpoint_{}_iter_*'.format(config['name'])
    checkpoint_dir = pathlib.Path(config['checkpoint_dir'])
    if checkpoint_dir.exists():
        for checkpoint_path in checkpoint_dir.iterdir():
            if (
                checkpoint_path.match(glob_pattern)
                and checkpoint_path.joinpath('COMPLETE').exists()
            ):
                iteration = max(int(checkpoint_path.name[-4:]), iteration)
    return iteration
