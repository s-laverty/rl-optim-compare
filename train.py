#!/usr/bin/env python

'''
This file defines the training algorithm for the deep Q network.

Created on 3/17/2023 by Steven Laverty (lavers@rpi.edu).
'''

import argparse
import enum
import json
import logging
import os
import pathlib
import typing
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from project.net import QNet

logging.basicConfig(
    format='%(asctime)s %(filename)s [%(levelname)s]: %(message)s',
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class Optimizer(str, enum.Enum):
    RMS_PROP = 'rms_prop'
    ADAM = 'adam'
    R_ADAM = 'r_adam'


class OptimConfig(typing.TypedDict, total=False):
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
    num_models: int
    q_lr: float
    q_gamma: float
    q_epsilon_max: float
    q_epsilon_min: float
    q_epsilon_decay: float
    q_replay_buf_len: int
    batch_size: int
    train_steps: int
    train_steps_target: int
    num_eval: int
    optimizers: dict[Optimizer, OptimConfig]


class Experience(typing.NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor


class Checkpoint(typing.NamedTuple):
    net_state_dict: dict
    optimizer_state_dict: dict
    replay_buffer: deque[Experience]
    train_reward_hist: typing.Sequence[float]
    eval_reward_hist: typing.Sequence[float]


def save_checkpoint(
    file: str | bytes | os.PathLike | typing.BinaryIO,
    checkpoint: Checkpoint,
) -> None:
    if isinstance(file, (str, bytes, os.PathLike)):
        dirname = os.path.dirname(file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    torch.save(checkpoint, file)


def load_checkpoint(
    file: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> Checkpoint:
    return torch.load(file, map_location='cpu')


def checkpoint_path(config: Config, iteration: int) -> pathlib.Path:
    name = 'checkpoint_{}_iter_{:04d}'.format(config['name'], iteration)
    return pathlib.Path(config['checkpoint_dir']).joinpath(name)


def latest_checkpoint_iteration(config: Config) -> int:
    iteration = -1
    glob_pattern = 'checkpoint_{}_iter_*.pt'.format(config['name'])
    num_checkpoints = len(config['optimizers']) * config['num_models']
    for checkpoint_path in pathlib.Path(config['checkpoint_dir']).iterdir():
        if (
            checkpoint_path.match(glob_pattern)
            and (
                sum(1 for path in checkpoint_path.iterdir() if path.is_file())
                == num_checkpoints
            )
        ):
            iteration = max(int(checkpoint_path.name[-4:]), iteration)
    return iteration


def train_net(
    config: Config,
    optimizer: Optimizer,
    rank: int,
    seed: int,
    iteration: int,
    num_iter: int = 1,
) -> None:
    pass

def init_optim(
    optimizer: Optimizer,
    config: OptimConfig,
    params: typing.Iterable[torch.nn.Parameter],
) -> optim.Optimizer:
    match optimizer:
        case [RMS_PROP]:
            return optim.RMSprop(
                params,
                config.get('lr'),
                config.get('rms_alpha'),
                config.get('eps'),
                config.get('weight_decay'),
                config.get('rms_momentum'),
                config.get('rms_centered'),
            )
        case [ADAM]:
            return optim.Adam(
                params,
                config.get('lr'),
                config.get('adam_betas'),
                config.get('eps'),
                config.get('weight_decay'),
                config.get('adam_amsgrad'),
            )
        case [R_ADAM]:
            return optim.RAdam(
                params,
                config.get('lr'),
                config.get('adam_betas'),
                config.get('eps'),
                config.get('weight_decay'),
            )
        case _:
            raise ValueError('Optimizer type {} not recognized'.format(optimizer))


class Args(argparse.Namespace):
    config_file: str
    num_iterations: int
    start_iteration: int
    parallel: bool


if __name__ == '__main__':
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file',
        type=str,
        help='Configuration JSON file used for training.',
    )
    parser.add_argument(
        'num_iterations',
        nargs='?',
        default=1,
        type=int,
        help='Number of training iterations (%(default)s by default).',
    )
    parser.add_argument(
        '-i',
        '--start-iteration',
        default=0,
        type=int,
        help='Iteration number to resume training from (%(metavar)s=%(default)s by default). To resume training automatically from the latest checkpoint, use %(metavar)s=-1.',
        metavar='N',
    )
    parser.add_argument(
        '-p',
        '--parallel',
        action='store_true',
        help='Run deep Q learning algorithms in parallel (multiprocessing) rather than in series. Requires more RAM.'
    )
    args = parser.parse_args(namespace=Args)

    # Parse config file
    config = Config(**json.load(args.config_file))

    # Find the latest checkpoint if the start iteration is zero
    if args.start_iteration == -1:
        args.start_iteration = latest_checkpoint_iteration(config) + 1

    # Begin training