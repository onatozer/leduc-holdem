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
    eval_env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': 0,
        }
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    
    agent = CFR(env = env)
    agent.load()  # If we have saved model, we first load the model

    # Evaluate CFR against random
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    with Logger(args.log_dir) as logger:
        for i in range(args.num_rounds):
            logger.log_performance(
                    i,
                    tournament(
                        eval_env,
                        args.num_eval_games
                    )[0]
                )
            csv_path, fig_path = logger.csv_path, logger.fig_path
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2_000,
    )
    parser.add_argument(
        '--num_rounds',
        type=int,
        default= 10,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_cfr_result/',
    )

    args = parser.parse_args()

    train(args)
    