''' An example of solve Leduc Hold'em with CFR (chance sampling)
'''
import os
import argparse

import rlcard


from cfr import CFR


from rlcard.agents import (
    RandomAgent
)


from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)

import numpy as np
import random


def train(args):
    # Make environments, CFR only supports Leduc Holdem
    env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    agent = CFR(env=env)
    agent.load()  # If we have saved model, we first load the model

    # Start training
    for i in range(args.epochs):
        agent.train()  # Perform training for one epoch or iteration
        if i % args.save_every == 0:
            agent.save()
            print(f'Model saved at epoch {i}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default= 100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_cfr_result/',
    )

    args = parser.parse_args()

    train(args)
    