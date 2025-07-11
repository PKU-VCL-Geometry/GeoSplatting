from __future__ import annotations

import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

from huggingface_hub import hf_hub_download

from rfstudio.engine.task import Task


@dataclass
class BaseDownloader(ABC, Task):

    save_dir: Path = Path('data')
    scene: str = 'all'
    use_proxy: bool = False
    exists: Literal['override', 'abort', 'skip'] = 'skip'
    manually_specify: Optional[Path] = None

    @property
    @abstractmethod
    def hf_id(self) -> str:
        ...

    @property
    @abstractmethod
    def files(self) -> Dict[str, Path]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def process(self, source: Path, target: Path, mutable: bool) -> None:
        ...

    def run(self) -> None:
        target_path = self.save_dir / self.name

        if self.manually_specify is not None:
            assert self.manually_specify.is_dir()
            for subfoler, filename in self.files.items():
                if self.scene not in ['all', subfoler]:
                    continue
                if (target_path / subfoler).exists():
                    if self.exists == 'skip':
                        continue
                    if self.exists == 'abort':
                        raise RuntimeError(f"There is an existing directory: {target_path}")
                    if self.exists == 'override':
                        shutil.rmtree(target_path / subfoler)
                    else:
                        raise ValueError(self.exists)
                self.process(self.manually_specify / filename, target_path, False)
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for subfoler, filename in self.files.items():
                if self.scene not in ['all', subfoler]:
                    continue
                if (target_path / subfoler).exists():
                    if self.exists == 'skip':
                        continue
                    if self.exists == 'abort':
                        raise RuntimeError(f"There is an existing directory: {target_path}")
                    if self.exists == 'override':
                        shutil.rmtree(target_path / subfoler)
                    else:
                        raise ValueError(self.exists)
                hf_hub_download(
                    self.hf_id,
                    str(filename),
                    repo_type='dataset',
                    local_dir=tmpdir,
                    endpoint='https://hf-mirror.com' if self.use_proxy else None,
                )
                if not (tmpdir / filename).exists():
                    return
                self.process(tmpdir / filename, target_path, True)
