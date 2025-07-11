from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, TypeVar

import torch

from rfstudio.data import BaseDataset
from rfstudio.nn import Module
from rfstudio.trainer import BaseTrainer
from rfstudio.ui import console
from rfstudio.utils.pretty import P
from rfstudio.visualization._optimization_visualizer import OptimizationVisualizer

from .experiment import Experiment
from .task import Task

T = TypeVar('T', bound='TrainTask')


@dataclass
class TrainTask(Task):

    dataset: BaseDataset = ...

    model: Module = ...

    experiment: Experiment = ...

    trainer: BaseTrainer = ...

    viser: OptimizationVisualizer = OptimizationVisualizer()

    def __setup__(self) -> None:
        super().__setup__()
        self.model.to(self.device)
        self.dataset.to(self.device)

    @property
    def ckpt_path(self) -> Path:
        return self.experiment.base_path / "ckpts"

    @property
    def script_path(self) -> Path:
        return self.experiment.base_path / "task.py"

    def run(self) -> None:

        assert self.trainer.batch_size % self.trainer.num_accums_per_batch == 0
        self.save_as_script(self.script_path)

        with console.status(P@'Loading dataset from {self.dataset.path} ...'):
            loader_train = self.dataset.get_train_iter(self.trainer.batch_size // self.trainer.num_accums_per_batch)
            if self.trainer.num_steps_per_val:
                loader_val = self.dataset.get_val_iter(1)

        with console.screen() as handle:
            handle.set_layout(
                handle.cols[3, 1](
                    handle.plot['train-loss'],
                    (
                        handle.rows(handle.table['train-metrics'], handle.table['val-metrics'])
                        if self.trainer.num_steps_per_val
                        else handle.table['train-metrics']
                    )
                )
            )
            handle.progress['training'].update(curr=0, total=self.trainer.num_steps)
            handle.sync()

            self.viser.setup(self.trainer.num_steps)
            optimizers = self.trainer.setup(self.model, self.dataset)
            if self.trainer.detect_anomaly:
                torch.set_anomaly_enabled(True)
            for step in range(1, self.trainer.num_steps + 1):
                optimizers.zero_grad()
                self.model.train(True)
                self.trainer.before_update(self.model, optimizers=optimizers, curr_step=step)

                visual = self.trainer.num_steps_per_vis and step % self.trainer.num_steps_per_vis == 0
                camera = self.viser.get_camera(step)
                for accum in range(self.trainer.num_accums_per_batch):
                    with torch.autocast(device_type=self.device_type, enabled=self.trainer.mixed_precision):
                        inputs, gt_outputs, indices = next(loader_train)
                        train_loss, train_metrics, visualization = self.trainer.step(
                            self.model,
                            inputs,
                            gt_outputs,
                            indices=indices,
                            training=True,
                            visual=visual and accum == 0
                        )
                        if self.trainer.detect_anomaly:
                            assert train_loss.isfinite()
                        if visual and accum == 0:
                            self.experiment.dump_image('train', index=step, image=visualization)
                    if accum == 0 and camera is not None:
                        with torch.no_grad():
                            visualization = self.trainer.visualize(self.model, camera.to(self.model.device))
                            self.experiment.dump_image('vis', index=step, image=visualization)
                    optimizers.backward(train_loss)

                handle.plot['train-loss'].update(
                    x=step,
                    y=train_loss.item(),
                )
                handle.table['train-metrics'].update(**train_metrics)

                self.trainer.after_backward(self.model)
                optimizers.step()
                self.trainer.after_update(self.model, optimizers=optimizers, curr_step=step)

                if self.trainer.num_steps_per_save and step % self.trainer.num_steps_per_save == 0:
                    self.save_checkpoint(step)

                if self.trainer.num_steps_per_val and step % self.trainer.num_steps_per_val == 0:
                    self.model.train(False)
                    with torch.no_grad():
                        inputs, gt_outputs, indices = next(loader_val)
                        _, val_metrics, visualization = self.trainer.step(
                            self.model,
                            inputs,
                            gt_outputs,
                            indices=indices,
                            training=False,
                            visual=True
                        )
                        self.experiment.dump_image('val', index=step, image=visualization)
                        self.experiment.log(P@'Val Metrics: {val_metrics}')
                        handle.table['val-metrics'].update(**val_metrics)

                handle.progress['training'].update(curr=step, total=self.trainer.num_steps)
                handle.sync()

            self.save_checkpoint(step)
            handle.sync(force=True)

            if self.trainer.after_train:
                self.trainer.after_train(self.model, self.dataset)

            if self.trainer.full_test_after_train:
                self.model.train(False)
                with torch.no_grad():
                    test_metrics = {}
                    test_step = 0
                    for inputs, gt_outputs, indices in self.dataset.get_test_iter(batch_size=1):
                        _, metrics, visualization = self.trainer.step(
                            self.model,
                            inputs,
                            gt_outputs,
                            indices=indices,
                            training=False,
                            visual=True
                        )
                        test_step += 1
                        self.experiment.dump_image('test', index=indices[0], image=visualization)
                        for key, val in metrics.items():
                            test_metrics.setdefault(key, []).append(val)
                        handle.progress['testing'].update(
                            curr=test_step,
                            total=self.dataset.get_size(split='test'),
                        )
                        handle.sync()

                    test_metrics = { key: sum(val) / len(val) for key, val in test_metrics.items() }
                    self.experiment.log(P@'Test Metrics: {test_metrics}')

            if self.trainer.hold_after_train:
                handle.hold('Finished. Ctrl+C to exit.')

    def save_checkpoint(self, step: int) -> None:
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
        ckpt_file_path = self.ckpt_path / f'{step:010d}.ckpt'
        torch.save(self.model.state_dict(), ckpt_file_path)

    def load_checkpoint(self, *, step: Optional[int] = None) -> Optional[int]:
        if not self.ckpt_path.exists():
            return
        if step is None:
            for filename in self.ckpt_path.glob('*.ckpt'):
                step_from_filename = int(filename.stem.rsplit('.', 1)[0])
                step = step_from_filename if step is None else max(step, step_from_filename)
            if step is None:
                return
        ckpt_file_path = self.ckpt_path / f'{step:010d}.ckpt'
        if not ckpt_file_path.exists():
            return
        self.model.load_state_dict(torch.load(ckpt_file_path, map_location='cpu'))
        return step

    @classmethod
    @torch.no_grad()
    def load_from_script(cls: Type[T], script_path: Path, *, step: Optional[int] = None) -> T:
        train_task = super().load_from_script(script_path)
        assert train_task.load_checkpoint(step=step) is not None
        return train_task
