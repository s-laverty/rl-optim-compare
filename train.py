#!/usr/bin/env python

'''
This file defines the training algorithm for the deep Q network.

Created on 3/17/2023 by Steven Laverty (lavers@rpi.edu).
'''

import argparse
import enum
import itertools
import json
import logging
import multiprocessing as mp
import os
import pathlib
import random
import time
import typing
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch_wrapper import TorchWrapper

from net import QNet

logging.basicConfig(
    format='%(asctime)s %(filename)s [%(levelname)s]: %(message)s',
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

OBS_SHAPE = 4
NUM_ACTIONS = 2


class OptimType(str, enum.Enum):
    RMS_PROP = 'rms_prop'
    ADAM = 'adam'
    R_ADAM = 'r_adam'
    ADAM_W = 'adam_w'


class OptimConfig(typing.TypedDict, total=False):
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
    device: torch.device = 'cpu',
) -> Checkpoint:
    return torch.load(file, map_location=device)


def checkpoint_path(config: Config, iteration: int) -> pathlib.Path:
    name = 'checkpoint_{}_iter_{:04d}'.format(config['name'], iteration)
    return pathlib.Path(config['checkpoint_dir']).joinpath(name)


def checkpoint_name(rank: int, optim_rank: int) -> pathlib.Path:
    return pathlib.Path('{}_{}.pt'.format(rank, optim_rank))


def latest_checkpoint_iteration(config: Config) -> int:
    iteration = -1
    glob_pattern = 'checkpoint_{}_iter_*.pt'.format(config['name'])
    checkpoint_dir = pathlib.Path(config['checkpoint_dir'])
    if checkpoint_dir.exists():
        for checkpoint_path in checkpoint_dir.iterdir():
            if (
                checkpoint_path.match(glob_pattern)
                and checkpoint_path.joinpath('COMPLETE').exists()
            ):
                iteration = max(int(checkpoint_path.name[-4:]), iteration)
    return iteration


def init_optim(
    config: OptimConfig,
    net: QNet,
) -> optim.Optimizer:
    match config['optim_type']:
        case [RMS_PROP]:
            return optim.RMSprop(
                net.parameters(),
                config.get('lr'),
                config.get('rms_alpha'),
                config.get('eps'),
                config.get('weight_decay'),
                config.get('rms_momentum'),
                config.get('rms_centered'),
            )
        case [ADAM]:
            return optim.Adam(
                net.parameters(),
                config.get('lr'),
                config.get('adam_betas'),
                config.get('eps'),
                config.get('weight_decay'),
                config.get('adam_amsgrad'),
            )
        case [R_ADAM]:
            return optim.RAdam(
                net.parameters(),
                config.get('lr'),
                config.get('adam_betas'),
                config.get('eps'),
                config.get('weight_decay'),
            )
        case _:
            raise ValueError(
                'Optimizer type {} not recognized'.format(config['optim_type']))


def epsilon_greedy(
    config: Config,
    steps: int,
    net: QNet,
    obs: torch.Tensor,
) -> int:
    epsilon = config['q_epsilon_min'] + (
        config['q_epsilon_decay']**steps
        * (config['q_epsilon_max'] - config['q_epsilon_min'])
    )
    if random.random() < epsilon:
        return random.choice(range(NUM_ACTIONS))
    net.eval()
    with torch.no_grad():
        q_values = net(obs)
    return torch.argmax(q_values)


def train_net(
    config: Config,
    policy_net: QNet,
    target_net: QNet,
    optimizer: optim.Optimizer,
    replay_buffer: deque[Experience],
    device: torch.device | None = None,
) -> None:
    policy_net.train()
    target_net.eval()

    # Get batch for training from replay buffer.
    states, actions, rewards, next_states = zip(*random.sample(
        replay_buffer,
        config['batch_size'],
    )),
    states = torch.stack(states)
    if device is not None:
        states = states.to(device)
    actions = states.new_tensor(actions, torch.long).unsqueeze(0)
    rewards = states.new_tensor(rewards)
    next_states_mask = states.new_tensor(
        tuple(s is not None for s in next_states),
        torch.bool,
    )
    next_states = torch.stack(tuple(s for s in next_states if s is not None))
    if device is not None:
        next_states = next_states.to(device)

    # Backpropagation.
    pred_q = policy_net(states).gather(1, actions).squeeze(1)
    next_v = torch.zeros_like(pred_q)
    with torch.no_grad():
        next_v[next_states_mask] = target_net(next_states).max(1)[0]
    target_q = next_v * config['q_gamma'] + rewards
    criterion = nn.SmoothL1Loss()
    loss = criterion(pred_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Soft update target net parameters.
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in target_net_state_dict:
        target_net_state_dict[key] += config['q_target_update_tau'] * (
            policy_net_state_dict[key] - target_net_state_dict[key]
        )


def eval_net(
    config: Config,
    net: QNet,
    env: gym.Env,
) -> float:
    net.eval()
    with torch.no_grad():
        total_rewards = []
        for _ in range(config['num_eval']):
            obs, _ = env.reset()
            total_reward = 0
            is_term = is_trunc = False
            while not is_term and not is_trunc:
                action = torch.argmax(net(obs)).item()
                obs, reward, is_term, is_trunc, _ = env.step(action)
                total_reward += reward
            total_rewards.append(total_reward)
        return sum(total_rewards) / len(total_rewards)


def deep_q(
    config: Config,
    rank: int,
    optim_rank: int,
    start_iteration: int,
    device: torch.device,
    env_seed: int,
    num_iter: int = 1,
    *,
    map_replay_buffer: bool = True,
    mp_barrier: mp.Barrier | None = None,
) -> None:
    name = checkpoint_name(rank, optim_rank)
    # Load previous checkpoint.
    (
        num_steps,
        policy_net_state_dict,
        target_net_state_dict,
        optimizer_state_dict,
        replay_buffer,
        eval_hist,
    ) = load_checkpoint(
        checkpoint_path(config, start_iteration - 1).joinpath(name),
        device if map_replay_buffer else 'cpu',
    )

    # Initialize neural nets.
    policy_net = QNet(OBS_SHAPE, NUM_ACTIONS)
    target_net = QNet(OBS_SHAPE, NUM_ACTIONS)
    policy_net.load_state_dict(policy_net_state_dict)
    target_net.load_state_dict(target_net_state_dict)
    if not map_replay_buffer:
        policy_net.to(device)
        target_net.to(device)
    optimizer = init_optim(config['optimizers'][optim_rank], policy_net)
    optimizer.load_state_dict(optimizer_state_dict)

    # Initialize environment.
    train_env = TorchWrapper(
        gym.make('CartPole-v1'),
        device=device if map_replay_buffer else 'cpu',
    )
    eval_env = TorchWrapper(
        gym.make('CartPole-v1'),
        device=device,
        batch_dim=True,
    )

    # Set training counter (neural net update very config['train_steps']).
    train_step_counter = num_steps % config['train_steps']
    if train_step_counter == 0:
        train_step_counter = config['train_steps']

    # Train for num_iter checkpoint iterations.
    obs, _ = train_env.reset(seed=env_seed)
    for checkpoint_iteration in range(
        start_iteration,
        start_iteration + num_iter,
    ):
        for _ in range(config['checkpoint_steps']):
            # Take an epsilon-greedy action, append to replay buffer.
            action = epsilon_greedy(config, num_steps, policy_net, obs)
            next_obs, reward, is_term, is_trunc, _ = train_env.step(action)
            replay_buffer.append(Experience(
                obs,
                action,
                reward,
                next_obs if not is_term else None,
            ))
            obs = next_obs
            if is_term or is_trunc:
                obs, _ = train_env.reset()
            num_steps += 1

            # Update policy (and target) every config['train_steps'].
            train_step_counter -= 1
            if train_step_counter == 0:
                train_step_counter = config['train_steps']
                train_net(
                    config,
                    policy_net,
                    target_net,
                    optimizer,
                    replay_buffer,
                    device if not map_replay_buffer else None,
                )
                eval_hist.append(eval_net(config, policy_net, eval_env))
        # Save new checkpoint.
        save_checkpoint(
            checkpoint_path(config, checkpoint_iteration).joinpath(name),
            Checkpoint(
                num_steps,
                policy_net.state_dict(),
                target_net.state_dict(),
                optimizer.state_dict(),
                replay_buffer,
                eval_hist,
            ),
        )

        # Communicate checkpoint completion to parent process.
        if mp_barrier is not None:
            mp_barrier.wait()


def init(
    config: Config,
    rank: int,
    device: torch.device,
) -> None:
    # All optimizers start with the same initial nets.
    policy_net = QNet(OBS_SHAPE, NUM_ACTIONS).to(device)
    target_net_state_dict = QNet(OBS_SHAPE, NUM_ACTIONS).state_dict()
    initial_eval = eval_net(
        config,
        policy_net,
        TorchWrapper(
            gym.make('CartPole-v1'),
            device=device,
            batch_dim=True,
        ),
    )

    # Create checkpoints for each optimizer type.
    path = checkpoint_path(config, 0)
    for optim_rank, optim_config in enumerate(config['optimizers']):
        optimizer_state_dict = (
            init_optim(optim_config, policy_net).state_dict()
        )
        replay_buffer = deque(maxlen=config['q_replay_buf_len'])
        save_checkpoint(
            path.joinpath(checkpoint_name(rank, optim_rank)),
            Checkpoint(
                0,
                policy_net.state_dict(),
                target_net_state_dict,
                optimizer_state_dict,
                replay_buffer,
                [initial_eval],
            )
        )


class Args(argparse.Namespace):
    config_file: str
    num_iterations: int
    start_iteration: int
    parallel: bool
    # clobber: bool
    remove_checkpoints: bool


def rr_assign(
    func: typing.Callable,
    args: list,
) -> list[mp.Process]:
    devices = (
        (torch.device('cuda', i) for i in range(torch.cuda.device_count()))
        if torch.cuda.is_available()
        else (torch.device('cpu'),)
    )
    processes = []
    for args_i, device in zip(args, itertools.cycle(devices)):
        proc = mp.Process(
            target=func,
            args=(device, *args_i),
        )
        proc.start()
        processes.append(proc)
    return processes


if __name__ == '__main__':
    # Parse CLI arguments.
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
        '-s',
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
    # parser.add_argument(
    #     '-c',
    #     '--clobber',
    #     action='store_true',
    #     help='Do not check for existing checkpoints. Overwrite them if they exist.',
    # )
    parser.add_argument(
        '-r',
        '--remove-checkpoints',
        action='store_true',
        help='Remove old checkpoints as newer checkpoints are saved.',
    )
    args = parser.parse_args(namespace=Args)

    # Parse config file.
    logger.debug('Parsing config file')
    config = Config(**json.load(args.config_file))

    # Find the latest checkpoint if the start iteration is -1.
    if args.start_iteration == -1:
        logger.debug('Starting after most recent checkpoint',
                     args.start_iteration)
        args.start_iteration = latest_checkpoint_iteration(config) + 1

    # Determine CUDA availability.
    if args.parallel:
        devices = (
            itertools.cycle(
                torch.device('cuda', i)
                for i in range(torch.cuda.device_count())
            )
            if torch.cuda.is_available()
            else itertools.repeat(torch.device('cpu'))
        )
    else:
        device = (
            torch.device('cuda')
            if torch.cuda.is_available()
            else torch.device('cpu')
        )

    # Initialize if the start iteration is 0.
    if args.start_iteration == 0:
        args.start_iteration += 1
        logger.info('Creating initial models for configuration %s',
                    config['name'])
        if args.parallel:
            # Initialize ranks in parallel.
            workers: list[mp.Process] = []
            for rank, device in zip(
                range(config['num_models']),
                itertools.cycle(devices)
            ):
                proc = mp.Process(
                    target=init,
                    name='init-worker-{}'.format(rank),
                    args=(config, rank, device),
                )
                proc.start()
                workers.append(proc)
            for proc in workers:
                proc.join()
                proc.close()
        else:
            # Initialize ranks sequentially.
            for rank in range(config['num_models']):
                init(config, rank, device)
        checkpoint_path(config, 0).joinpath('COMPLETE').touch()

    # Begin training.
    if args.parallel:
        # Initialize training workers.
        barrier = mp.Barrier(
            1 + config['num_models'] * len(config['optimizers']),
        )
        workers: list[mp.Process] = []
        devices_iter = iter(devices)
        for rank in range(config['num_models']):
            seed = time.time_ns()
            for optim_rank in range(len(config['optimizers'])):
                proc = mp.Process(
                    target=deep_q,
                    name='train-worker-{}-{}'.format(rank, optim_rank),
                    args=(
                        config,
                        rank,
                        optim_rank,
                        args.start_iteration,
                        next(devices_iter),
                        seed,
                        args.num_iterations,
                    ),
                    kwargs={
                        'mp_barrier': barrier,
                    },
                )
                proc.start()
                workers.append(proc)
    
    # Train for args.num_iterations
    for checkpoint_iteration in range(
        args.start_iteration,
        args.start_iteration + args.num_iterations,
    ):
        if args.parallel:
            # Wait for training to complete in parallel.
            barrier.wait()
        else:
            # Train each net sequentially.
            for rank in range(config['num_models']):
                seed = time.time_ns()
                for optim_rank in range(len(config['optimizers'])):
                    deep_q(
                        config,
                        rank,
                        optim_rank,
                        checkpoint_iteration,
                        device,
                        seed,
                    )
        # Mark checkpoint as completed.
        checkpoint_path(config, checkpoint_iteration) \
            .joinpath('COMPLETE').touch()
        if args.remove_checkpoints:
            prev_path = checkpoint_path(config, checkpoint_iteration - 1)
            if prev_path.exists():
                os.removedirs(prev_path)
    
    if args.parallel:
        # Terminate training workers.
        for proc in workers:
            proc.join()
            proc.close()
