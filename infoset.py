from typing import List, cast, Dict, NewType
import numpy as np


Player = NewType('Player', int)
Action = NewType('Action', int)

class InfoSet:
    """
    Information Set I_i
    """

    def __init__(self, key: bytes, legal_actions: List[Action]):
        """
        Initialize the InfoSet with a key and the legal actions available at this state.
        """
        self.key = key
        self.legal_actions = legal_actions
        self.regret = {a: 0.0 for a in legal_actions}
        self.cumulative_strategy = {a: 0.0 for a in legal_actions}
        self.strategy = {}
        self.calculate_strategy()

    def actions(self) -> List[Action]:
        """
        Return the list of legal actions.
        """
        return self.legal_actions

    def calculate_strategy(self):
        """
        Calculate current strategy using regret matching.
        """
        positive_regrets = {a: max(r, 0) for a, r in self.regret.items()}
        regret_sum = sum(positive_regrets.values())
        if regret_sum > 0:
            self.strategy = {a: positive_regrets[a] / regret_sum for a in self.legal_actions}
        else:
            count = len(self.legal_actions)
            self.strategy = {a: 1.0 / count for a in self.legal_actions}

    def get_average_strategy(self):
        """
        Get average strategy based on cumulative strategy.
        """
        strategy_sum = sum(self.cumulative_strategy.values())
        if strategy_sum > 0:
            return {a: self.cumulative_strategy[a] / strategy_sum for a in self.legal_actions}
        else:
            count = len(self.legal_actions)
            return {a: 1.0 / count for a in self.legal_actions}

    def to_dict(self):
        """
        Save the information set to a dictionary.
        """
        return {
            'key': self.key,
            'regret': self.regret,
            'average_strategy': self.cumulative_strategy,
        }

    def load_dict(self, data: Dict[tuple, any]):
        """
        Load data from a saved dictionary.
        """
        self.regret = data['regret']
        self.cumulative_strategy = data['average_strategy']
        self.calculate_strategy()

    def __repr__(self):
        """
        Human-readable string representation of the InfoSet.
        """
        total = sum(self.cumulative_strategy.values())
        if total > 0:
            strategy = {a: self.cumulative_strategy[a] / total for a in self.legal_actions}
        else:
            count = len(self.legal_actions)
            strategy = {a: 1.0 / count for a in self.legal_actions}
        return f'InfoSet(key={self.key}, strategy={strategy})'

