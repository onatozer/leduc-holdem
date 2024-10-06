import numpy as np
from typing import cast
from infoset import InfoSet, Player, Action
import os
import pickle
import random
from rlcard.utils.utils import *



class CFR:
   
    def __init__(self, *,
                 n_players: int = 2,
                 env
                 ):
        """
        * `create_new_history` creates a new empty history
        * `epochs` is the number of iterations to train on $T$
        * `n_players` is the number of players
        """
        self.n_players = n_players
        self.env = env
        self.use_raw = False
        # A dictionary for $\mathcal{I}$ set of all information sets
        # self.create_new_history = create_new_history
        self.info_sets = {}

    def get_info_set_key(self, player_id):
        """
        Generates a unique key for the information set based on the player's observations.

        Args:
            player_id (int): The ID of the player for whom the key is being generated.

        Returns:
            bytes: A byte representation of the player's observation, used as a key for the info set.
        """
        state = self.env.get_state(player_id)
        return self._state_abstraction(state)

    #TODO: Write your own abstraction here:
    def _state_abstraction(self, state):
        ...


    def _get_info_set(self, info_set_key, legal_actions):
        """
        Returns the information set I for the current player at a given state.
        """
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InfoSet(info_set_key, legal_actions)
        return self.info_sets[info_set_key]



    #TODO: Reimplement the walk_tree function, but instead of using the history class, use the built-in class env variable
    def walk_tree(self, i: Player, pi_i: float, pi_neg_i: float) -> float:
        ...

    
    def eval_step(self, state):
        edited_state = self._state_abstraction(state)
        legal_actions = list(state['legal_actions'].keys())

        if edited_state not in self.info_sets:
            # Assign uniform probabilities if the info set is not found
            action_probs = [1.0 / len(legal_actions) for _ in legal_actions]
        else:
            info_set = self.info_sets[edited_state]
            average_strategy = info_set.get_average_strategy()
            # Get probabilities for legal actions
            action_probs = [average_strategy.get(a, 0.0) for a in legal_actions]

        # Normalize action_probs to sum to 1
        sum_probs = sum(action_probs)
        if sum_probs > 0:
            action_probs = [p / sum_probs for p in action_probs]
        else:
            # If the sum is zero (all probabilities are zero), assign uniform probabilities
            action_probs = [1.0 / len(legal_actions) for _ in legal_actions]

        # Now, action_probs should sum to 1
        action = np.random.choice(legal_actions, p=action_probs)
        return action, action_probs


    
    #Ok, I think we did it, can come back later for some better information logging
    def train(self, epochs = 1):
        """
        ### Iteratively update $\textcolor{lightgreen}{\sigma^t(I)(a)}$

        This updates the strategies for $T$ iterations.
        """

        # Loop for `epochs` times
        for t in range(epochs):
            for i in range(self.n_players):
                self.env.reset()
                self.walk_tree(cast(Player, i), 1, 1)     

        #     # Save checkpoints every $1,000$ iterations
        #     if (t + 1) % 1_000 == 0:
        #         experiment.save_checkpoint()

        # # Print the information sets
        # logger.inspect(self.info_sets)

    def save(self, model_path = './cfr_model'):
        ''' Save model
        '''
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        cfr_policy = open(os.path.join(model_path, 'policy.pkl'),'wb')
        pickle.dump(self.info_sets, cfr_policy)
        cfr_policy.close()

    def load(self, model_path = './cfr_model'):
        ''' Load model
        '''
        if not os.path.exists(model_path):
            print(f'No model found at {model_path}')
            return

        policy_file = open(os.path.join(model_path, 'policy.pkl'),'rb')
        self.info_sets = pickle.load(policy_file)
        policy_file.close()

       
