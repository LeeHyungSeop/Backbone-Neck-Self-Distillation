from collections import OrderedDict, abc as container_abcs
from itertools import chain, islice
import operator

import torch
from torch.nn.modules.container import Module
from torch._jit_internal import _copy_to_script_wrapper

from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
from typing_extensions import Self
from typing_extensions import deprecated

T = TypeVar('T', bound=Module)

class Sequential(Module):
    _modules: Dict[str, Module]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx) -> T:  # type: ignore[misc, type-var]
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f'index {idx} is out of range')
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[slice, int]) -> Union['Sequential', T]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    def __add__(self, other) -> 'Sequential':
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError('add operator supports only objects '
                             f'of Sequential class, but {str(type(other))} is given.')

    def pop(self, key: Union[int, slice]) -> Module:
        v = self[key]
        del self[key]
        return v

    def __iadd__(self, other) -> Self:
        if isinstance(other, Sequential):
            offset = len(self)
            for i, module in enumerate(other):
                self.add_module(str(i + offset), module)
            return self
        else:
            raise ValueError('add operator supports only objects '
                             f'of Sequential class, but {str(type(other))} is given.')

    def __mul__(self, other: int) -> 'Sequential':
        if not isinstance(other, int):
            raise TypeError(f"unsupported operand type(s) for *: {type(self)} and {type(other)}")
        elif (other <= 0):
            raise ValueError(f"Non-positive multiplication factor {other} for {type(self)}")
        else:
            combined = Sequential()
            offset = 0
            for _ in range(other):
                for module in self:
                    combined.add_module(str(offset), module)
                    offset += 1
            return combined

    def __rmul__(self, other: int) -> 'Sequential':
        return self.__mul__(other)

    def __imul__(self, other: int) -> Self:
        if not isinstance(other, int):
            raise TypeError(f"unsupported operand type(s) for *: {type(self)} and {type(other)}")
        elif (other <= 0):
            raise ValueError(f"Non-positive multiplication factor {other} for {type(self)}")
        else:
            len_original = len(self)
            offset = len(self)
            for _ in range(other - 1):
                for i in range(len_original):
                    self.add_module(str(i + offset), self._modules[str(i)])
                offset += len_original
            return self

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    def forward(self, input):
        outs = []
        for module in self:
            input = module(input)
            outs.append(input)
            # print(f"module : {module}")
            # print(f"input : {input.shape}")
        return [outs[3], outs[5], outs[7]]

    def append(self, module: Module) -> 'Sequential':
        r"""Append a given module to the end.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self


    def insert(self, index: int, module: Module) -> 'Sequential':
        if not isinstance(module, Module):
            raise AssertionError(
                f'module should be of type: {Module}')
        n = len(self._modules)
        if not (-n <= index <= n):
            raise IndexError(
                f'Index out of range: {index}')
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential) -> 'Sequential':
        for layer in sequential:
            self.append(layer)
        return self