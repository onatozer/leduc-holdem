from typing import List, cast, Dict, Callable
import numpy as np
from cfr import History as _History, InfoSet as _InfoSet, Action, Player

from pettingzoo.classic import leduc_holdem_v4 as leduc


#TODO: I'm thinking that we define history class so that it accepts instance of 
# Infoset class definition and cfr class so that it accepts instance of history definition
class InfoSet(_InfoSet):
    """
    ## [Information set](../index.html#InfoSet)
    """

    @staticmethod
    def from_dict(data: Dict[str, any]) -> 'InfoSet':
        """Does not support save/load"""
        pass

    def actions(self) -> List[Action]:
        """
        Return the list of actions. Terminal states are handled by `History` class.
        """
        #last 4 entries of the tuple constitute the 'action mask' -> don't know if I like this though, 
        # has to return list of unique numbers to index though regret dictionaries

        action_mask = self.key[-4:]
        valid_actions = []
        for i, a in enumerate(action_mask):
            if a == 1:
                valid_actions.append(cast(Action,i))
        
        return valid_actions

    def __repr__(self):
        """
        Human readable string representation - it gives the betting probability
        """
        total = sum(self.cumulative_strategy.values())
        total = max(total, 1e-6)
        bet = self.cumulative_strategy[cast(Action, 'b')] / total
        return f'{bet * 100: .1f}%'


class History(_History):
    """
    ## [History](../index.html#History)

    This defines when a game ends, calculates the utility and sample chance events (dealing cards).

    The history is stored in a string:

    * First two characters are the cards dealt to player 1 and player 2
    * The third character is the action by the first player
    * Fourth character is the action by the second player
    """ 


    def __init__(self, seed = None, actions = None):
        """
        Initialize with a given history string
        """
        super().__init__()
        #Should this be a shallow copy ??
        self.actions = actions.copy() if actions else []
        self.seed = seed if seed is not None else np.random.randint(1_000)

        # history = leduc.env(render_mode = 'ansi')
        # history.reset(seed=np.random.randint(1_000))
        # self.history = history

        #We need to have a player counter for this to work, maybe not anymore ??

    #TODO: How do I manage closing of the environments, could be making the code significantly slower
    def get_env(self):
        """
        Creates the environment and puts it in the state that corresponds to the given actions taken
        """
        env = leduc.env(render_mode='ansi')
        env.reset(seed=self.seed)
        for action in self.actions:
            env.step(action)

        return env
    

    def is_terminal(self):
        """
        Whether the history is terminal (game over).
        """
        
        env = self.get_env()
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            return True
        else:
            return False

    def terminal_utility(self, i: Player) -> float:
        """
        Get the terminal utility for player $i$
        """
        # If $i$ is Player 1
        env = self.get_env()
        observation, reward, termination, truncation, info = env.last()
        if i == self.player():
            return reward
        else:
            return -1*reward
        

    
    def __add__(self, other: Action):
        """
        Add an action to the history and return a new history
        Treat this as simply taking step in env with action other
        """
        #god bless GPT
        return History(actions=self.actions + [other], seed=self.seed)


    def player(self) -> Player:
        """
        Current player
        """
        return cast(Player, len(self.actions) % 2)

    #Not gonna worry about this now
    # def __repr__(self):
    #     """
    #     Human readable representation
    #     """
    #     return repr(self.history)

    #NOTE: important
    #TODO: Change so that output is a byte string so that indexing into dictionary is mroe efficient
    def info_set_key(self) -> tuple:
        """
        Information set key for the current history.
        This is a string of actions only visible to the current player.
        """
    
        #Ok, looks simple enough, just return literally all information that isn't the opponent hand + actions you can perform
        env = self.get_env()
        observation, reward, termination, truncation, info = env.last()

        return tuple(observation['observation']) + tuple(observation['action_mask'])
        

    def new_info_set(self) -> InfoSet:
        # Create a new information set object
        return InfoSet(self.info_set_key())
