from typing import List, Any
import numpy as np


class Hyperparameter():
    ''' Class that store an hyperparameter and provides easy sampling and mutating based on priority.
    Args:
        name: name of the entry, should be unique
        params: the parameter, any python object
        choices: a list of python object
        priority: a number representing the priority when evoloving/mutating parameters, 0 = constant.
    '''
    def __init__(self, name: str, params: Any, choices: List[Any], priority: int) -> None:
        self.name = name
        self.params = params
        self.choices = choices
        self.priority = priority

    def mutate(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def check(self):
        raise NotImplementedError


class HyperparameterDict(dict):
    ''' Class that store key values pairs <name - hyperparamaters>
    '''
    def __setitem__(self, key, value):
        assert isinstance(value, Hyperparameter), print(f'Value is not an Hyperparameter')
        super().__setitem__(key, value)

    def __getitem__(self, key):
        return super().__getitem__(key).params

    def get(self, key, default=None):
        return super().get(key, default).params