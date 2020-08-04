from typing import List, Any
import numbers


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

    def check(self) -> bool:
        if self.priority > 0:
            if self.params in self.choices:
                return True
            else:
                # For numeral parameters we additionally check boundaries of the choices.
                if isinstance(self.params, numbers.Number) and (self.params >= self.choices[0] and self.params <= self.choices[-1]):
                    return True
                return False
        return True


class HyperparameterDict(dict):
    ''' Class that store key values pairs <name - hyperparamaters>
    '''
    def __setitem__(self, key, value):
        assert isinstance(value, Hyperparameter), print('Value is not an Hyperparameter')
        super().__setitem__(key, value)

    def __getitem__(self, key):
        return super().__getitem__(key).params

    def get(self, key, default=None):
        return super().get(key, default).params

    def check(self):
        '''Verifies some part of the config, mainly the presence of some keys and checks all sub members
        '''

        # Check all params are inside of choices of each HP
        for key in super().keys():
            item = super().__getitem__(key)
            assert(item.check()), print(f'config entry {key} with value {item.params} was not in {item.choices}')

        # Check minimum stuff is inside the config
        # TODO
        return True

    def __str__(self):
        ret_str = ''
        for key in super().keys():
            item = self.__getitem__(key)
            ret_str += f'{key}: {item}\n'
        return ret_str
