import numpy as np
from typing import NewType, Dict, List, Callable, cast
import os
import pickle

# A player $i \in N$ where $N$ is the set of players
Player = NewType('Player', int)
# Action $a$, $A(h) = {a: (h, a) \in H}$ where $h \in H$ is a non-terminal [history](#History)

Action = NewType('Action', int)


class History:
  

    def is_terminal(self):
        """
        Whether it's a terminal history; i.e. game over.
        $h \in Z$
        """
        raise NotImplementedError()

    def terminal_utility(self, i: Player) -> float:
        """
        <a id="terminal_utility"></a>
        Utility of player $i$ for a terminal history.
        $u_i(h)$ where $h \in Z$
        """
        raise NotImplementedError()

    def player(self) -> Player:
        """
        Get current player, denoted by $P(h)$, where $P$ is known as **Player function**.

        If $P(h) = c$ it means that current event is a chance $c$ event.
        Something like dealing cards, or opening common cards in poker.
        """
        raise NotImplementedError()

    def is_chance(self) -> bool:
        """
        Whether the next step is a chance step; something like dealing a new card.
        $P(h) = c$
        """
        raise NotImplementedError()

    def sample_chance(self) -> Action:
        """
        Sample a chance when $P(h) = c$.
        """
        raise NotImplementedError()

    def __add__(self, action: Action):
        """
        Add an action to the history.
        """
        raise NotImplementedError()

    def info_set_key(self) -> str:
        """
        Get [information set](#InfoSet) for the current player
        """
        raise NotImplementedError

    def new_info_set(self) -> 'InfoSet':
        """
        Create a new [information set](#InfoSet) for the current player
        """
        raise NotImplementedError()

    def __repr__(self):
        """
        Human readable representation
        """
        raise NotImplementedError()

class InfoSet:
    """
    <a id="InfoSet"></a>

    ## Information Set $I_i$
    """

    #unique identifier for the information set
    key: tuple

    strategy: Dict[Action, float]
    regret: Dict[Action, float]
    
    cumulative_strategy: Dict[Action, float]

    def __init__(self, key: tuple):
        """
        Initialize
        """
        self.key = key
        self.regret = {a: 0 for a in self.actions()}
        self.cumulative_strategy = {a: 0 for a in self.actions()}
        self.calculate_strategy()

    def actions(self) -> List[Action]:
        """
        Actions $A(I_i)$
        """
        raise NotImplementedError()

    #why does ts exist??
    @staticmethod
    def from_dict(data: Dict[str, any]) -> 'InfoSet':
        """
        Load information set from a saved dictionary
        """
        raise NotImplementedError()

    def to_dict(self):
        """
        Save the information set to a dictionary
        """
        return {
            'key': self.key,
            'regret': self.regret,
            'average_strategy': self.cumulative_strategy,
        }

    def load_dict(self, data: Dict[tuple, any]):
        """
        Load data from a saved dictionary
        """
        self.regret = data['regret']
        self.cumulative_strategy = data['average_strategy']
        self.calculate_strategy()

    def calculate_strategy(self):
        """
        ## Calculate strategy

        Calculate current strategy using [regret matching](#RegretMatching).
        """
        regret = {a: max(r, 0) for a, r in self.regret.items()}
        regret_sum = sum(regret.values())
        if regret_sum > 0:
            
            self.strategy = {a: r / regret_sum for a, r in regret.items()}
        # Otherwise,
        else:
            count = len(list(a for a in self.regret))
            
            self.strategy = {a: 1 / count for a, r in regret.items()}

    def get_average_strategy(self):
        """
        ## Get average strategy

        """
        cum_strategy = {a: self.cumulative_strategy.get(a, 0.) for a in self.actions()}
       
        strategy_sum = sum(cum_strategy.values())
        if strategy_sum > 0:
           
            return {a: s / strategy_sum for a, s in cum_strategy.items()}
        # Otherwise,
        else:
            count = len(list(a for a in cum_strategy))
            
            return {a: 1 / count for a, r in cum_strategy.items()}

    def __repr__(self):
        """
        Human readable representation
        """
        raise NotImplementedError()

class CFR:
   
    #might have to make this a tuple :-(
    info_sets: Dict[tuple, InfoSet]

    def __init__(self, *,
                 n_players: int = 2,
                 create_new_history: Callable[[], History]):
        """
        * `create_new_history` creates a new empty history
        * `epochs` is the number of iterations to train on $T$
        * `n_players` is the number of players
        """
        self.n_players = n_players
        # A dictionary for $\mathcal{I}$ set of all information sets
        self.create_new_history = create_new_history
        self.info_sets = {}


    def _get_info_set(self, h: History):
        """
        Returns the information set $I$ of the current player for a given history $h$
        """
        info_set_key = h.info_set_key()
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = h.new_info_set()
        return self.info_sets[info_set_key]

    def walk_tree(self, h: History, i: Player, pi_1: float, pi_2: float) -> float:
        if h.is_terminal():
           return h.terminal_utility(i)
        
        
        I = self._get_info_set(h)

        v = 0
        v_a = {}

        for a in I.actions():
            #player 1 -> 0, player 2 -> 1
            if h.player() == 0:
                v_a[a] = self.walk_tree(h + a, i, pi_1*I.strategy[a], pi_2)

            elif h.player() == 1:
                v_a[a] = self.walk_tree(h + a, i, pi_1, pi_2*I.strategy[a])

            v += v_a[a]*I.strategy[a]

        if(h.player() == i):

            if(h.player() == 0):
                pi_i = pi_1
                pi_neg_i = pi_2
            else:
                pi_i = pi_2
                pi_neg_i = pi_1

            for a in I.actions():
                #maybe update cumulative strategy here ??
                I.regret[a] += pi_neg_i*(v_a[a] - v)
                I.cumulative_strategy[a] += pi_i*I.strategy[a]

            I.calculate_strategy()

        return v


    #TODO:
    #Ok, I think we did it, can come back later for some better information logging

    def train(self, epochs):
        """
        ### Iteratively update $\textcolor{lightgreen}{\sigma^t(I)(a)}$

        This updates the strategies for $T$ iterations.
        """

        # Loop for `epochs` times
        for t in range(epochs):
            for i in range(self.n_players):
                self.walk_tree(self.create_new_history(), cast(Player, i), 1, 1)     

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

       
