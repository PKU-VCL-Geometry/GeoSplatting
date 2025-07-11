from __future__ import annotations

import cProfile
import pdb
import random
import sys
from dataclasses import dataclass, fields, is_dataclass, replace
from multiprocessing import Process
from pathlib import Path
from types import MethodType
from typing import (
    Annotated,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import torch
import tyro

from rfstudio.utils.dataclass import (
    DataclassInstance,
    dump_dataclass_as_str,
    has_no_missing_field,
    load_dataclass,
)
from rfstudio.utils.hook import inject
from rfstudio.utils.pretty import pretty_traceback


def _recursive_check_dataclass(data: Task) -> Task:
    assert is_dataclass(data)
    changes = {}
    for f in fields(data):
        value = getattr(data, f.name)
        if value is Ellipsis:
            changes[f.name] = tyro.MISSING
        elif is_dataclass(value):
            changes[f.name] = _recursive_check_dataclass(value)
    return replace(data, **changes)

def _subrun(self: Task) -> None:
    pr: cProfile.Profile = None
    try:
        assert has_no_missing_field(self), "Missing field is not allowed"
        torch.set_float32_matmul_precision('high')
        torch.set_printoptions(
            precision=3,
            threshold=16,
        )
        if self.device.type != 'cpu':
            torch.cuda.set_device(self.device)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        self.run = MethodType(self.__class__.run, self)
        if self.profiling is not None:
            pr = cProfile.Profile()
            pr.enable()
        _setup(self)
        self.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if not isinstance(self, Task):
            self = Task()
        if self.traceback_pretty:
            pretty_traceback(show_locals=True, max_frames=self.traceback_frames)
            if self.auto_breakpoint:
                pdb.post_mortem()
            else:
                sys.exit(-1)
        elif self.auto_breakpoint:
            sys.stderr.write(f'{e.__class__.__name__}: {e}\n')
            pdb.post_mortem()
        else:
            raise
    finally:
        if pr is not None:
            pr.disable()
            pr.dump_stats(str(self.profiling))


@runtime_checkable
class Component(Protocol):

    def __setup__(self) -> None:
        ...


T = TypeVar('T', bound='Task')


@dataclass
class Task:

    """
    TODO
    """

    seed: Optional[int] = None

    cuda: Optional[int] = None

    traceback_pretty: bool = True

    traceback_frames: int = 8

    profiling: Optional[Path] = None

    auto_breakpoint: bool = True

    @property
    def device(self) -> torch.device:
        return torch.device('cpu') if self.cuda is None else torch.device(f'cuda:{self.cuda}')

    @property
    def device_type(self) -> Literal['cpu', 'cuda']:
        return 'cpu' if self.cuda is None else 'cuda'

    def __setup__(self) -> None:
        pass

    def run(self) -> None:
        raise NotImplementedError

    def join(self, timeout: Optional[float] = None) -> None:
        process = Process(target=_subrun, args=(self,))
        process.start()
        process.join(timeout=timeout)

    def detach(self) -> None:
        process = Process(target=_subrun, args=(self,))
        process.start()

    @classmethod
    def __new__(cls: Type[T], *args, **kwargs) -> T:
        instance = super().__new__(cls)
        inject(instance.run, _entrypoint)
        return instance

    def save_as_script(self, script_path: Path) -> None:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(dump_dataclass_as_str(self))
            f.write(
                """\nif __name__ == '__main__':\n"""
                """    dumped.run()\n""",
            )

    @classmethod
    def load_from_script(cls: Type[T], script_path: Path) -> T:
        assert script_path.exists()
        task = load_dataclass(script_path)
        assert isinstance(task, cls)
        task.run = MethodType(cls.run, task)
        _setup(task)
        return task


class TaskGroup:

    def __init__(self, **tasks: Task) -> None:
        self.tasks = { key.replace('_', '-'): _recursive_check_dataclass(value) for key, value in tasks.items() }

    def run(self) -> None:
        _entrypoint(self)


def _entrypoint(t: Union[Task, TaskGroup]) -> None:
    self: Task = None
    pr: cProfile.Profile = None
    try:
        if isinstance(t, TaskGroup):
            parsed = Union.__getitem__(
                tuple(
                    Annotated[
                        tyro.conf.AvoidSubcommands[v.__class__],
                        tyro.conf.subcommand(k, default=v, description="", prefix_name=True),
                    ] for (k, v) in t.tasks.items()
                )
            )
            self = tyro.cli(parsed)
        elif isinstance(t, Task):
            self = tyro.cli(tyro.conf.AvoidSubcommands[t.__class__], default=_recursive_check_dataclass(t))
        else:
            raise TypeError(
                "Invalid type for `t`: expect Union[Task, TaskGroup], "
                f"but receive {t.__class__.__name__} instead"
            )
        assert has_no_missing_field(self), "Missing field is not allowed"
        torch.multiprocessing.set_start_method('spawn')
        torch.set_float32_matmul_precision('high')
        torch.set_printoptions(
            precision=3,
            threshold=16,
        )
        if self.device.type != 'cpu':
            torch.cuda.set_device(self.device)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        self.run = MethodType(self.__class__.run, self)
        if self.profiling is not None:
            pr = cProfile.Profile()
            pr.enable()
        _setup(self)
        self.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if not isinstance(self, Task):
            self = Task()
        if self.traceback_pretty:
            pretty_traceback(show_locals=True, max_frames=self.traceback_frames)
            if self.auto_breakpoint:
                pdb.post_mortem()
            else:
                sys.exit(-1)
        elif self.auto_breakpoint:
            sys.stderr.write(f'{e.__class__.__name__}: {e}\n')
            pdb.post_mortem()
        else:
            raise
    finally:
        if pr is not None:
            pr.disable()
            pr.dump_stats(str(self.profiling))


def _setup(x: Union[DataclassInstance, Component]) -> None:
    if is_dataclass(x):
        for f in fields(x):
            value = getattr(x, f.name)
            _setup(value)
    if isinstance(x, Component):
        x.__setup__()
