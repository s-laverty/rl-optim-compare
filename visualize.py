#!/usr/bin/env python

'''
This file defines utilities for visualizing results.

Created on 3/17/2023 by Steven Laverty (lavers@rpi.edu).
'''

import argparse
import json
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from utils import (Config, checkpoint_name, checkpoint_path,
                   latest_checkpoint_iteration, load_checkpoint)

logging.basicConfig(
    format='%(asctime)s %(filename)s [%(levelname)s]: %(message)s',
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def plot_loss(
    config: Config,
    iteration: int,
    smoothing: bool,
    ranks: list[int],
):
    if not ranks:
        ranks = range(len(config['optimizers']))
    path = checkpoint_path(config, iteration)
    if not path.joinpath('COMPLETE').exists():
        raise ValueError(
            'Invalid checkpoint. Checkpoint {} is incomplete'.format(iteration))
    # shape=(len(optimizers),num_models,num_train_iter)
    loss_history = np.asarray([
        [
            load_checkpoint(path.joinpath(
                checkpoint_name(rank, optim_rank)))[5]
            for rank in range(config['num_models'])
        ]
        for optim_rank in ranks
    ])
    mean_loss = loss_history.mean(axis=1)
    confidence_bound_lower = np.quantile(loss_history, 0.25, axis=1)
    confidence_bound_upper = np.quantile(loss_history, 0.75, axis=1)
    iterations = np.arange(loss_history.shape[-1]) * config['train_steps']
    if smoothing:
        # Get moving averages
        window_size = 19
        mean_loss = [np.convolve(x, np.ones(
            window_size), 'valid') / window_size for x in mean_loss]
        confidence_bound_lower = [np.convolve(x, np.ones(
            window_size), 'valid') / window_size for x in confidence_bound_lower]
        confidence_bound_upper = [np.convolve(x, np.ones(
            window_size), 'valid') / window_size for x in confidence_bound_upper]
        iterations = np.arange(
            window_size // 2, (loss_history.shape[-1] - (window_size - 1) // 2)) * config['train_steps']
    plt.figure(figsize=(7, 5))
    for mean_loss_i, lower_i, upper_i, name in zip(
        mean_loss,
        confidence_bound_lower,
        confidence_bound_upper,
        (config['optimizers'][optim_rank]['name'] for optim_rank in ranks)
    ):
        plt.fill_between(iterations, lower_i, upper_i, alpha=0.2)
        plt.plot(iterations, mean_loss_i, label=name)
    plt.legend()
    plt.title('Cart-pole performance by optimization algorithm')
    plt.xlabel('Total steps')
    plt.ylabel('Average duration')
    plt.show()


if __name__ == '__main__':
    # Parse CLI arguments.
    class Args(argparse.Namespace):
        config_file: str
        checkpoint_iteration: int
        smoothing: bool
        ranks: list[int]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file',
        type=str,
        help='Configuration JSON file used for training.',
    )
    parser.add_argument(
        'checkpoint_iteration',
        nargs='?',
        default=-1,
        type=int,
        help='Checkpoint iteration to visualize (%(default)s by default). To load the latest checkpoint, use %(metavar)s=-1.',
    )
    parser.add_argument(
        '-s',
        '--smoothing',
        action='store_true',
        help='Use data-smoothing in the plot.',
    )
    parser.add_argument(
        '-r',
        '--ranks',
        action='extend',
        nargs='+',
        type=int,
        help='Choose a specific subset of optimizers to plot.'
    )
    args = parser.parse_args(namespace=Args)

    # Parse config file.
    logger.debug('Parsing config file')
    with open(args.config_file) as f:
        config = Config(**json.load(f))

    # Find the latest checkpoint if the checkpoint iteration is -1.
    if args.checkpoint_iteration == -1:
        args.checkpoint_iteration = latest_checkpoint_iteration(config)
        logger.debug('Using most recent checkpoint (iteration %d)',
                     args.checkpoint_iteration)

    # Plot the loss history
    plot_loss(config, args.checkpoint_iteration, args.smoothing, args.ranks)
