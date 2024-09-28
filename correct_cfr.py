''' An example of solve Leduc Hold'em with CFR (chance sampling)
'''
import os
import argparse

import rlcard

# from cfr_agent import CFRAgent
from cfr import CFR
from define_abstractions import History


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

class CFRAgent(CFR):
    ''' Implement CFR (chance sampling) algorithm
    '''

    def __init__(self, env, model_path='./cfr_model'):

        super().__init__(create_new_history = History)
        self.use_raw = False
        self.env = env
        self.model_path = model_path

        #debug variables
        self.random_actions = 0
        self.agent_actions = 0

    def _show_info_state_probs(self):
        for i in self.info_sets.keys():
            print(f'key: {self.info_sets[i].key}\n cumulative strategy: {self.info_sets[i].cumulative_strategy}')

    def eval_step(self, state):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
            info (dict): A dictionary containing information
        '''

        legal_actions = [0,0,0,0]

        for i in range(4):
            if i in state['legal_actions']:
                legal_actions[i] = 1


        obs_array = np.array(state['obs'], dtype= np.uint8)
        action_mask = np.array(state['obs'], dtype = np.uint8)

        edited_state = np.concatenate((obs_array,action_mask))
        # print(f'dict index before byte: {edited_state}')

        edited_state = edited_state.tobytes()


        # print(f'num info set keys {len(self.info_sets.keys())}')

        #index into info_set with edited_state, and cumulative_strategy should give us our probability distribution

        #If the state doesn't have an entry in our info_set dict, then we just take the averages of the possible options
        if edited_state not in self.info_sets.keys():
            self._show_info_state_probs
            # print(edited_state)

            action_list = []
            for i, value in enumerate(legal_actions):
                if value == 1:
                    action_list.append(i)
            
            action = random.choice(action_list)
            #NOTE: this might not be of the correct length
            self.random_actions +=1 
            probabilities = [1/len(action_list) for i in action_list]

        else:
            info_set = self.info_sets[edited_state]
            average_strategy = info_set.get_average_strategy()
            probabilities = list(average_strategy.values())

            #Do I need to use numpy for this ??
            self.agent_actions += 1
            action = np.random.choice(list(average_strategy.keys()), p = probabilities)

        return action, probabilities

    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        return state['obs'].tostring(), list(state['legal_actions'].keys())


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

    # Initilize CFR Agent
    agent = CFRAgent(
        env,
        os.path.join(
            args.log_dir,
            'cfr_model',
        ),
    )
    agent.load()  # If we have saved model, we first load the model

    # Evaluate CFR against random
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    # Start training
    #NOTE: ????
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            #run the 'walk trees function for 100 epochs
            agent.train(epochs = 100)
            # agent.save()
            # Evaluate the performance. Play with Random agents.
            if episode % args.evaluate_every == 0:
                print('\rIteration {}'.format(episode), end='')
                print(f'random actions taken {agent.random_actions}. Agent actions taken {agent.agent_actions}')
                agent.save() # Save model
                logger.log_performance(
                    episode,
                    tournament(
                        eval_env,
                        args.num_eval_games
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot the learning curve
    # plot_curve(csv_path, fig_path, 'cfr')

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
        default=100,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_cfr_result/',
    )

    args = parser.parse_args()

    train(args)
    