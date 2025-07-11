from __future__ import annotations

import itertools
import warnings
import weakref
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Tensor, nn

from rfstudio.utils.hook import wrap_hook


class _HashHelper:
    def __init__(self, parent: Module) -> None:
        self.parent = weakref.ref(parent)


T = TypeVar('T', bound=Union['Module', nn.Module])


def _setup_recursively(self: Module) -> None:
    for child in self.children():
        if isinstance(child, Module):
            _setup_recursively(child)
        else:
            assert isinstance(child, nn.Module)
    self.__setup__()


def _setup_updated_children(self: Union[Module, nn.Module], promise: Callable[[], None]) -> None:
    children_before = { Module._make_hashable_(child) for child in self.children() }
    promise()
    children_after = { Module._make_hashable_(child) for child in self.children() }
    for child in (children_after - children_before):
        if isinstance(child, _HashHelper):
            _setup_recursively(child.parent())
        else:
            assert isinstance(child, nn.Module)


@dataclass
class Module:

    """TODO"""

    @classmethod
    def _make_hashable_(cls, t: Any) -> Any:
        if isinstance(t, Module):
            return t._hash_helper
        return t

    @classmethod
    def _remove_all_fields(cls) -> None:
        for f in fields(cls):
            if hasattr(cls, f.name):
                p = cls
                while p is not Module:
                    try:
                        type.__delattr__(p, f.name)
                    except AttributeError:
                        p = cls.__base__
                        continue
                    break
        type.__setattr__(cls, '_remove_all_fields', lambda: None)

    @classmethod
    def __new__(cls: Type[Module], *args, **kwargs) -> Module:
        instance = object.__new__(cls)
        cls._remove_all_fields()
        object.__setattr__(instance, '_version', 1)
        object.__setattr__(instance, '_training', True)
        object.__setattr__(instance, '_device', torch.zeros(0).device)
        object.__setattr__(instance, '_parameters', OrderedDict())
        object.__setattr__(instance, '_buffers', OrderedDict())
        object.__setattr__(instance, '_non_persistent_buffers_set', set())
        object.__setattr__(instance, '_modules', OrderedDict())
        object.__setattr__(instance, '_hash_helper', _HashHelper(instance))
        wrap_hook(instance.__setup__, _setup_updated_children)
        return instance

    def __setup__(self) -> None:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> Any:
        return self(*args, **kwargs)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def training(self) -> bool:
        return self._training

    def _check_assign(self, name: str, container: OrderedDict[str, Any]) -> bool:
        if not isinstance(name, str):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format(torch.typename(name)))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in container:
            raise KeyError("attribute '{}' already exists".format(name))
        return True

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        """
        Adds a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """

        if self._check_assign(name, self._buffers):
            if tensor is not None and not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    "cannot assign '{}' object to buffer '{}' "
                    "(torch Tensor or None required)".format(torch.typename(tensor), name)
                )
            else:
                self._buffers[name] = tensor
                if persistent:
                    self._non_persistent_buffers_set.discard(name)
                else:
                    self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[nn.Parameter]) -> None:
        """
        Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """

        self._check_assign(name, self._parameters)

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, nn.Parameter):
            raise TypeError(
                "cannot assign '{}' object to parameter '{}' "
                "(torch.nn.Parameter or None required)".format(torch.typename(param), name)
            )
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name)
            )
        else:
            self._parameters[name] = param

    def add_module(self, name: str, module: Union[Module, nn.Module, None]) -> None:
        """
        Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        _modules: OrderedDict[str, Union[Module, nn.Module, None]] = self.__dict__['_modules']
        if not isinstance(module, (Module, nn.Module)) and module is not None:
            raise TypeError("{} is not a Module subclass".format(torch.typename(module)))
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(torch.typename(name)))
        elif hasattr(self, name) and name not in _modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\", got: {}".format(name))
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        _modules[name] = module

    def register_module(self, name: str, module: Union[Module, nn.Module, None]) -> None:
        """Alias for :func:`add_module`."""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> Union[Module, nn.Module]:
        """
        Returns the submodule given by ``target`` if it exists,
        otherwise throws an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: Union[Module, nn.Module] = self

        for item in atoms:

            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, (Module, nn.Module)):
                raise AttributeError("`" + item + "` is not "
                                     "an nn.Module")

        return mod

    def get_parameter(self, target: str) -> nn.Parameter:
        """
        Returns the parameter given by ``target`` if it exists,
        otherwise throws an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the Parameter
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Parameter: The Parameter referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Parameter``
        """
        module_path, _, param_name = target.rpartition(".")

        mod: Union[Module, nn.Module] = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(mod._get_name() + " has no attribute `" + param_name + "`")

        param: nn.Parameter = getattr(mod, param_name)

        if not isinstance(param, nn.Parameter):
            raise AttributeError("`" + param_name + "` is not an nn.Parameter")

        return param

    def get_buffer(self, target: str) -> Tensor:
        """
        Returns the buffer given by ``target`` if it exists,
        otherwise throws an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the buffer
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.Tensor: The buffer referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not a
                buffer
        """
        module_path, _, buffer_name = target.rpartition(".")

        mod: Union[Module, nn.Module] = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(mod._get_name() + " has no attribute `" + buffer_name + "`")

        buffer: torch.Tensor = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

    def _apply(self: T, fn: Callable[[Any], Any]) -> T:
        for module in self.children():
            module._apply(fn)

        def compute_should_use_set_data(tensor: Tensor, tensor_applied: Tensor) -> bool:
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                # If the new tensor has compatible tensor type as the existing tensor,
                # the current behavior is to change the tensor in-place using `.data =`,
                # and the future behavior is to overwrite the existing tensor. However,
                # changing the current behavior is a BC-breaking change, and we want it
                # to happen in future releases. So for now we introduce the
                # `torch.__future__.get_overwrite_module_params_on_conversion()`
                # global flag to let the user control whether they want the future
                # behavior of overwriting the existing tensor or not.
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            return False

        for key, param in self._parameters.items():
            if param is None:
                continue
            # Tensors stored in modules are graph leaves, and we don't want to
            # track autograd history of `param_applied`, so we have to use
            # `with torch.no_grad():`
            with torch.no_grad():
                param_applied = fn(param)
            should_use_set_data = compute_should_use_set_data(param, param_applied)
            if should_use_set_data:
                param.data = param_applied
                out_param = param
            else:
                assert isinstance(param, nn.Parameter)
                assert param.is_leaf
                out_param = nn.Parameter(param_applied, param.requires_grad)
                self._parameters[key] = out_param

            if param.grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param.grad)
                should_use_set_data = compute_should_use_set_data(param.grad, grad_applied)
                if should_use_set_data:
                    assert out_param.grad is not None
                    out_param.grad.data = grad_applied
                else:
                    assert param.grad.is_leaf
                    out_param.grad = grad_applied.requires_grad_(param.grad.requires_grad)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def apply(self: T, fn: Callable[[Union[Module, nn.Module]], None]) -> T:
        """
        Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model
        (see also :ref:`nn-init-doc`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            >>> @torch.no_grad()
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )

        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        """
        Moves all model parameters and buffers to the GPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on GPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        self._device = torch.zeros(0).cuda(device).device
        return self._apply(lambda t: t.cuda(device))

    def cpu(self: T) -> T:
        """
        Moves all model parameters and buffers to the CPU.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        self._device = torch.device('cpu')
        return self._apply(lambda t: t.cpu())

    def type(self: T, dst_type: Union[torch.dtype, str]) -> T:
        """
        Casts all parameters and buffers to :attr:`dst_type`.

        .. note::
            This method modifies the module in-place.

        Args:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.type(dst_type))

    def float(self: T) -> T:
        """
        Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

    def double(self: T) -> T:
        """
        Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)

    def half(self: T) -> T:
        """
        Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

    def bfloat16(self: T) -> T:
        """
        Casts all floating point parameters and buffers to ``bfloat16`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.bfloat16() if t.is_floating_point() else t)

    def to_empty(self: T, *, device: Union[str, torch.device]) -> T:
        """
        Moves the parameters and buffers to the specified device without copying storage.

        Args:
            device (:class:`torch.device`): The desired device of the parameters
                and buffers in this module.

        Returns:
            Module: self
        """
        self._device = device
        return self._apply(lambda t: torch.empty_like(t, device=device))

    @overload
    def to(
        self: T,
        device: Optional[Union[int, torch.device]] = ...,
        dtype: Optional[Union[torch.dtype, str]] = ...,
        non_blocking: bool = ...
    ) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[torch.dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self: T, *args, **kwargs) -> T:
        r"""
        Moves and/or casts the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)
           :noindex:

        .. function:: to(dtype, non_blocking=False)
           :noindex:

        .. function:: to(tensor, non_blocking=False)
           :noindex:

        .. function:: to(memory_format=torch.channels_last)
           :noindex:

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point or complex :attr:`dtype`\ s. In addition, this method will
        only cast the floating point or complex parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            memory_format (:class:`torch.memory_format`): the desired memory
                format for 4D parameters and buffers in this module (keyword
                only argument)

        Returns:
            Module: self

        Examples::

            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> linear = nn.Linear(2, 2)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]])
            >>> linear.to(torch.double)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]], dtype=torch.float64)
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
            >>> gpu1 = torch.device("cuda:1")
            >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
            >>> cpu = torch.device("cpu")
            >>> linear.to(cpu)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16)

            >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.3741+0.j,  0.2382+0.j],
                    [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
            >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
            tensor([[0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

        """

        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None:
            Module._set_device_recursively(self, device)

        if dtype is not None:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise TypeError(
                    'nn.Module.to only accepts floating point or complex '
                    'dtypes, but got desired dtype={}'.format(dtype)
                )
            if dtype.is_complex:
                warnings.warn(
                    "Complex modules are a new feature under active development whose design may change, "
                    "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                    "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
                    "if a complex module does not work as expected."
                )

        def convert(t: Tensor) -> Tensor:
            if convert_to_format is not None and t.dim() in (4, 5):
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                    memory_format=convert_to_format
                )
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

        return self._apply(convert)

    @classmethod
    def _set_device_recursively(cls, target: Union[Module, nn.Module], device: torch.device) -> None:
        for module in target.children():
            Module._set_device_recursively(module, device)
        if isinstance(target, Module):
            target._device = device

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)

    def __getattr__(self, name: str) -> Any:
        _parameters = self.__dict__['_parameters']
        if name in _parameters:
            return _parameters[name]
        _buffers = self.__dict__['_buffers']
        if name in _buffers:
            return _buffers[name]
        modules = self.__dict__['_modules']
        if name in modules:
            return modules[name]
        if isinstance(getattr(self.__class__, name, None), property):
            return self.__getattribute__(name)
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __setattr__(self, name: str, value: Any) -> None:

        def remove_from(*dicts_or_sets) -> None:
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    elif isinstance(d, set):
                        d.discard(name)
                    else:
                        raise NotImplementedError

        params = self.__dict__['_parameters']
        if isinstance(value, nn.Parameter):
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif name in params:
            if value is not None:
                raise TypeError(
                    "cannot assign '{}' as parameter '{}' "
                    "(torch.nn.Parameter or None expected)".format(torch.typename(value), name)
                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__['_modules']
            if isinstance(value, (nn.Module, Module)):
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            elif name in modules:
                if value is not None:
                    raise TypeError(
                        "cannot assign '{}' as child module '{}' "
                        "(torch.nn.Module or None expected)".format(torch.typename(value), name)
                    )
                modules[name] = value
            else:
                buffers = self.__dict__['_buffers']
                if name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError(
                            "cannot assign '{}' as buffer '{}' "
                            "(torch.Tensor or None expected)".format(torch.typename(value), name)
                        )
                    buffers[name] = value
                else:
                    super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def get_extra_state(self) -> Any:
        """
        Returns any extra state to include in the module's state_dict.
        Implement this and a corresponding :func:`set_extra_state` for your module
        if you need to store extra state. This function is called when building the
        module's `state_dict()`.

        Note that extra state should be picklable to ensure working serialization
        of the state_dict. We only provide provide backwards compatibility guarantees
        for serializing Tensors; other objects may break backwards compatibility if
        their serialized pickled form changes.

        Returns:
            object: Any extra state to store in the module's state_dict
        """
        raise RuntimeError(
            "Reached a code path in Module.get_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug."
        )

    def set_extra_state(self, state: Any) -> NoReturn:
        """
        This function is called from :func:`load_state_dict` to handle any extra state
        found within the `state_dict`. Implement this function and a corresponding
        :func:`get_extra_state` for your module if you need to store extra state within its
        `state_dict`.

        Args:
            state (dict): Extra state from the `state_dict`
        """
        raise RuntimeError(
            "Reached a code path in Module.set_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug."
        )

    def _save_to_state_dict(self, destination: Dict, prefix: str, keep_vars: bool) -> None:
        """
        Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """

        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        extra_state_key = prefix + nn.modules.module._EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "get_extra_state", Module.get_extra_state) is not Module.get_extra_state:
            destination[extra_state_key] = self.get_extra_state()

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
    # back that same object. But if they pass nothing, an `OrderedDict` is created and returned.
    T_destination = TypeVar('T_destination', bound=Dict[str, Any])

    @overload
    def state_dict(self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination:
        ...

    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]:
        ...

    def state_dict(self, *, destination: Optional[Dict] = None, prefix: str = '',
                   keep_vars: bool = False) -> Dict[str, Any]:
        """
        Returns a dictionary containing references to the whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        .. note::
            The returned object is a shallow copy. It contains references
            to the module's parameters and buffers.

        .. warning::
            Currently ``state_dict()`` also accepts positional arguments for
            ``destination``, ``prefix`` and ``keep_vars`` in order. However,
            this is being deprecated and keyword arguments will be enforced in
            future releases.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): a prefix added to parameter and buffer
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> module.state_dict().keys()
            ['bias', 'weight']

        """

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        return destination

    def _load_from_state_dict(
        self,
        state_dict: Dict,
        prefix: str,
        local_metadata: Dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str]
    ) -> None:
        """
        Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """

        persistent_buffers = { k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set }
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = { k: v for k, v in local_name_params if v is not None }

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if not torch.overrides.is_tensor_like(input_param):
                    error_msgs.append(
                        'While copying the parameter named "{}", '
                        'expected torch.Tensor or Tensor-like object from checkpoint but '
                        'received {}'.format(key, type(input_param))
                    )
                    continue

                # This is used to avoid copying uninitialized parameters into
                # non-lazy modules, since they dont have the hook to do the checks
                # in such case, it will error when accessing the .shape attribute.
                is_param_lazy = nn.parameter.is_lazy(param)
                if is_param_lazy:
                    param.materialize(input_param.shape)
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append(
                        'size mismatch for {}: copying a param with shape {} from checkpoint, '
                        'the shape in current model is {}.'.format(key, input_param.shape, param.shape)
                    )
                    continue
                try:
                    with torch.no_grad():
                        param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append(
                        'While copying the parameter named "{}", '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}, '
                        'an exception occurred : {}.'.format(key, param.size(), input_param.size(), ex.args)
                    )
            elif strict:
                missing_keys.append(key)

        extra_state_key = prefix + nn.modules.module._EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "set_extra_state", Module.set_extra_state) is not Module.set_extra_state:
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
            elif strict:
                missing_keys.append(extra_state_key)
        elif strict and (extra_state_key in state_dict):
            unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) and key != extra_state_key:
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]           # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ) -> nn.modules.module._IncompatibleKeys:
        """
        Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError("Expected state_dict to be dict-like, got {}.".format(type(state_dict)))

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module: Union[Module, nn.Module], local_state_dict: Dict, prefix: str = '') -> None:
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    child_state_dict = { k: v for k, v in local_state_dict.items() if k.startswith(child_prefix) }
                    load(child, child_state_dict, child_prefix)

        load(self, state_dict)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)
                    )
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in missing_keys))
                )

        if len(error_msgs) > 0:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(self.__class__.__name__, "\n\t".join(error_msgs))
            )
        return nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)

    def _named_members(
        self,
        get_members_fn: Callable[[Union[nn.Module, Module]], Iterable[Tuple[str, Any]]],
        prefix: str = '',
        recurse: bool = True,
        remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Any]]:
        """Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or Module._make_hashable_(v) in memo:
                    continue
                if remove_duplicate:
                    memo.add(Module._make_hashable_(v))
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True,
                         remove_duplicate: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, param in self.named_parameters():
            >>>     if name in ['bias']:
            >>>         print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate
        )
        yield from gen

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """
        Returns an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = '', recurse: bool = True,
                      remove_duplicate: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """
        Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool, optional): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

        Yields:
            (str, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, buf in self.named_buffers():
            >>>     if name in ['running_var']:
            >>>         print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(), prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )
        yield from gen

    def children(self) -> Iterator[Union[Module, nn.Module]]:
        """
        Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, Union[Module, nn.Module]]]:
        """
        Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and Module._make_hashable_(module) not in memo:
                memo.add(Module._make_hashable_(module))
                yield name, module

    def modules(self) -> Iterator[Union[Module, nn.Module]]:
        """
        Returns an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            ...     print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self, memo: Optional[Set[Union[Module, nn.Module]]] = None, prefix: str = '', remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Union[Module, nn.Module]]]:
        """
        Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """

        if memo is None:
            memo = set()
        if Module._make_hashable_(self) not in memo:
            if remove_duplicate:
                memo.add(Module._make_hashable_(self))
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def train(self: T, mode: bool = True) -> T:
        """
        Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self._training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self: T) -> T:
        """
        Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        return self.train(False)

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        """
        Change if autograd should record operations on parameters in this
        module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.requires_grad_()` and several similar mechanisms that may be confused with it.

        Args:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        """
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Sets gradients of all model parameters to zero. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, '_is_replica', False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead."
            )

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def share_memory(self: T) -> T:
        """See :meth:`torch.Tensor.share_memory_`"""
        return self._apply(lambda t: t.share_memory_())

    def _get_name(self) -> str:
        return self.__class__.__name__

    def __dir__(self) -> List[str]:
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def _replicate_for_data_parallel(self: T) -> T:
        replica = self.__new__(type(self))
        replica.__dict__ = self.__dict__.copy()

        # replicas do not have parameters themselves, the replicas reference the original
        # module.
        replica._parameters = OrderedDict()
        replica._buffers = replica._buffers.copy()
        replica._modules = replica._modules.copy()
        replica._is_replica = True

        return replica


@dataclass
class ParameterModule(Module):

    shape: List[int] = ...

    init_value: float = 0.0

    def __setup__(self) -> None:
        self.params = nn.Parameter(torch.empty(self.shape).fill_(self.init_value))

    @classmethod
    def from_tensor(cls, t: Tensor) -> ParameterModule:
        module = ParameterModule(t.shape)
        module.params.data.copy_(t)
        return module
