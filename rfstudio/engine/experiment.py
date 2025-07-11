from __future__ import annotations

import atexit
import pathlib
import time
from dataclasses import dataclass

from torch import Tensor

from rfstudio.io import dump_float32_image
from rfstudio.utils.pretty import depretty


@dataclass
class Experiment:

    """
    TODO
    """

    name: str = ...
    """
    related to path for dumping experiment files: ${output_dir}/${name}/${timestamp}
    """

    output_dir: pathlib.Path = pathlib.Path('outputs')
    """
    related to path for dumping experiment files: ${output_dir}/${name}/${timestamp}
    """

    timestamp: str = None
    """
    related to path for dumping experiment files: ${output_dir}/${name}/${timestamp}.
    Use current timestamp when not specified.
    """

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self._logger = None

    @property
    def base_path(self) -> pathlib.Path:
        return self.output_dir / self.name / self.timestamp

    @property
    def log_path(self) -> pathlib.Path:
        return self.base_path / "log.txt"

    @property
    def dump_path(self) -> pathlib.Path:
        return self.base_path / "dump"

    def log(self, text: str) -> None:
        if self._logger is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            file = open(self.log_path, "a")
            atexit.register(lambda: file.close())
            self._logger = file
        lines = depretty(text).splitlines()
        time_str = time.strftime("[%Y-%m-%d %H:%M:%S] ")
        space_str = ' ' * len(time_str)
        self._logger.write(time_str + lines[0].rstrip() + '\n')
        for i in range(1, len(lines)):
            self._logger.write(space_str + lines[i].rstrip() + '\n')
        self._logger.flush()

    def dump_image(self, subfolder: str, *, index: int, image: Tensor, mkdir: bool = True) -> None:
        assert image.min().item() >= 0 and image.max().item() <= 1
        path = self.dump_path / subfolder
        if mkdir:
            path.mkdir(exist_ok=True, parents=True)
        assert path.exists()
        filename = path / f'{index:04d}.png'
        dump_float32_image(filename, image)
