from abc import abstractmethod
from itertools import chain

"""
All choices of hyperparameters are stored as a list of dict, e.g., [{'a': 1, 'b': 1}, {'a': 2, 'b': 1}]
Join generates all combinations of these choices
e.g.: [{'a': 1}], [{'b': 1}, {'b': 2}] -> [{'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
Union generates the union of these choices
e.g.: [{'a': 1}], [{'b': 1}, {'b': 2}] -> [{'a': 1}, {'b': 1}, {'b': 2}]
"""

def str_args(args):
    return ','.join(map(str, args))


class HParamSequence:
    @abstractmethod
    def __iter__(self):
        pass


class HParamNode(HParamSequence):
    def __init__(self, *args):
        self.subspaces = list(args)
        for subspace in self.subspaces:
            assert isinstance(subspace, HParamSequence), type(subspace)

    def __call__(self, *args):
        new = type(self)(*args)
        new.subspaces = self.subspaces + new.subspaces
        return new

    def __str__(self):
        return f'{type(self).__name__}({str_args(self.subspaces)})'


def grid_search_generator(grid):
    def rec(index, hparams):
        if index == len(grid):
            yield hparams.copy()
        else:
            choices = grid[index]
            for choice in choices:
                assert type(choice) == dict, choice
                new_hparams = hparams.copy()
                new_hparams.update(choice)
                yield from rec(index + 1, new_hparams)

    yield from rec(0, {})


class Join(HParamNode):
    def __iter__(self):
        return grid_search_generator(self.subspaces)


class Union(HParamNode):
    def __iter__(self):
        yield from chain(*self.subspaces)


class HP(HParamSequence):
    def __init__(self, **kwargs):
        assert len(kwargs) == 1
        key, values = next(iter(kwargs.items()))
        if type(values) != list:
            values = [values]
        self.key = key
        self.values = values

    def __iter__(self):
        return ({self.key: value} for value in self.values)

    def __str__(self):
        return f'{type(self).__name__}({self.key}={self.values})'
