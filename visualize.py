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

from utils import (Checkpoint, Config, Experience, checkpoint_name,
                   checkpoint_path, latest_checkpoint_iteration,
                   load_checkpoint)

logging.basicConfig(
    format='%(asctime)s %(filename)s [%(levelname)s]: %(message)s',
)
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def plot_loss(config: Config, iteration: int):
    path = checkpoint_path(config, iteration)
    print(path)
    if not path.joinpath('COMPLETE').exists():
        raise ValueError('Invalid checkpoint. Checkpoint {} is incomplete'.format(iteration))
    # shape=(len(optimizers),num_models,num_train_iter)
    loss_history = np.asarray([
        [
            load_checkpoint(path.joinpath(checkpoint_name(rank, optim_rank)))[5]
            for rank in range(config['num_models'])
        ]
        for optim_rank in range(len(config['optimizers']))
    ])
    mean_loss = loss_history.mean(axis=1)
    confidence_bound_lower = np.quantile(loss_history, 0.025, axis=1)
    confidence_bound_upper = np.quantile(loss_history, 0.975, axis=1)
    iterations = np.arange(loss_history.shape[-1]) * config['train_steps']
    plt.figure(figsize=(7,5))
    for mean_loss_i, lower_i, upper_i, name in zip(
        mean_loss,
        confidence_bound_lower,
        confidence_bound_upper,
        (optim_config['name'] for optim_config in config['optimizers'])
    ):
        plt.fill_between(iterations, lower_i, upper_i, alpha=0.2)
        plt.plot(iterations, mean_loss_i, label=name)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Parse CLI arguments.
    class Args(argparse.Namespace):
        config_file: str
        checkpoint_iteration: int
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
    args = parser.parse_args(namespace=Args)

    # Parse config file.
    logger.debug('Parsing config file')
    with open(args.config_file) as f:
        config = Config(**json.load(f))

    # Find the latest checkpoint if the checkpoint iteration is -1.
    if args.checkpoint_iteration == -1:
        args.checkpoint_iteration = latest_checkpoint_iteration(config)
        logger.debug('Using most recent checkpoint (iteration %d)', args.checkpoint_iteration)
    
    # Plot the loss history
    plot_loss(config, args.checkpoint_iteration)
