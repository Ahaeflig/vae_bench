from typing import List, Any, Dict
import numbers
import inspect


class Hyperparameter():
    
    ''' Class that store an hyperparameter and provides easy sampling and mutating based on priority.
    Args:
        name: name of the entry, should be unique
        params: the parameter, any python object
        choices: a list of python object
        inner_args: if the inner parameters contains mutable values they can be specified, enabling recursive search etc
        priority: a number representing the priority when evoloving/mutating parameters, 0 = constant.
    '''
    def __init__(self, name: str, params: Any, choices: List[Any], inner_args: Dict, priority: int) -> None:
        self.name = name
        self.params = params
        self.choices = choices
        self.inner_args = inner_args
        self.priority = priority

        # TODO check construction

    def get_value(self):
        if self.inner_args:
            # Get list of valid arguments for the function
            possible_args = inspect.signature(self.params).parameters.keys()
            # Generate args to pass to function
            args = {k: self.inner_args[k].get_value() for k in self.inner_args.keys() if k in possible_args}
            return self.params(**args)
        else:
            return self.params

    def get_inner_description(self):
        if self.inner_args:
            descr = ''
            for key in self.inner_args.keys():
                descr += f'    {key}: {self.inner_args[key].get_inner_description()}\n'
            return descr
        else:
            return str(self.params)

    # TODO not valid anymore correct with
    def check(self) -> bool:
        raise NotImplementedError
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
        return super().__getitem__(key).get_value()

    def get(self, key, default=None):
        return super().get(key, default).get_value()

    def check(self):
        '''Verifies some part of the config, mainly the presence of some keys and checks all sub members
        '''

        # Check all params are inside of choices of each HP
        for key in super().keys():
            item = super().__getitem__(key)
            assert(item.check()), print(f'config entry {key} with value {item.get_value()} was not in {item.choices}')
        return True

    def __str__(self):
        ret_str = ''
        for key in super().keys():
            item = self.__getitem__(key)
            ret_str += f'{key}: {item}\n'
            if super().__getitem__(key).inner_args:
              ret_str += f'{super().__getitem__(key).get_inner_description()}'
        return ret_str
