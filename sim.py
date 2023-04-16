#!/usr/bin/env python

'''
This file runs a simulation using the policy defined by an input file.

Created on 4/6/2023 by Steven Laverty (lavers@rpi.edu).
'''

import argparse
import json
import random

import gymnasium as gym
import numpy as np
import torch
from matplotlib import animation
from matplotlib import pyplot as plt

from net import QNet
from torch_wrapper import TorchWrapper
from utils import NUM_ACTIONS, OBS_SHAPE, Config, load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file',
        type=str,
        help='Configuration JSON file used for training.',
    )
    parser.add_argument(
        'checkpoint_file',
        type=str,
        help='Checkpoint file used for policy.',
    )
    parser.add_argument(
        '-b',
        '--best',
        action='store_true',
        help='If provided, use the best model from the provided checkpoint, rather than the most recent model.',
    )
    parser.add_argument(
        '-e',
        '--epsilon',
        type=float,
        help='If provided, use an epsilon-greedy policy with this epsilon',
    )
    parser.add_argument(
        '-n',
        '--num-steps',
        type=int,
        help='If provided, run the simulation for exactly this many steps. Reset if necessary.',
    )
    parser.add_argument(
        '-o',
        '--out-file',
        type=str,
        help='Output file to store animation.',
    )

    class Args(argparse.Namespace):
        config_file: str
        checkpoint_file: str
        best: bool
        epsilon: float | None
        num_steps: int | None
        out_file: str | None
    args = parser.parse_args(namespace=Args)

    # Parse config file.
    with open(args.config_file) as f:
        config = Config(**json.load(f))

    # Determine device availability
    device = (
        torch.device('cuda')
        if torch.cuda.is_available()
        else torch.device('cpu')
    )

    # Load checkpoint
    (
        _,
        policy_net_state_dict,
        _,
        _,
        _,
        eval_hist,
        best_state_dict,
    ) = load_checkpoint(args.checkpoint_file, device)

    # Initialize neural net.
    net = QNet(
        OBS_SHAPE,
        NUM_ACTIONS,
        num_layers=config['hidden_layers'],
        hidden_dim=config['hidden_dim'],
    )
    net.load_state_dict(
        best_state_dict if args.best else policy_net_state_dict,
    )

    # Initialize environment.
    env = TorchWrapper(
        gym.make(
            'CartPole-v1',
            render_mode='human' if not args.out_file else 'rgb_array_list',
        ),
        device=device,
        batch_dim=True,
    )

    # Run simulation
    frames = []
    net.eval()
    with torch.no_grad():
        obs, _ = env.reset()
        num_steps = 0
        complete = False
        while not complete:
            if args.epsilon and random.random() < args.epsilon:
                action = random.choice(range(NUM_ACTIONS))
            else:
                action = torch.argmax(net(obs)).item()
            obs, _, is_term, is_trunc, _ = env.step(action)
            num_steps += 1
            if args.num_steps:
                complete = num_steps == args.num_steps
                if is_term or is_trunc:
                    if args.out_file:
                        frames.extend(env.render())
                        frames.extend(10 * [255 * np.ones_like(frames[0])])
                    obs, _ = env.reset()
            else:
                complete = is_term or is_trunc

    # Save output video
    if args.out_file:
        frames.extend(env.render())
        # Create animation and save it.
        dpi = 72.0
        plt.figure(
            figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi),
            dpi=72.0,
        )
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i: int):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(
            plt.gcf(),
            animate,
            frames=len(frames),
            interval=50,
        )
        anim.save(
            args.out_file,
            writer='ffmpeg',
            fps=30,
        )
