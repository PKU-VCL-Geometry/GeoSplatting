from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from rfstudio.graphics import Cameras


@dataclass
class OptimizationVisualizer:

    center: Tuple[float, float, float] = (0, 0, 0)
    up: Literal['+y', '+z', 'disable'] = 'disable'
    spin_resolution: int = 4096
    hfov_degree: int = 40
    resolution: Tuple[int, int] = (800, 800)
    pitch_degree: int = 30
    radius: float = 3.2

    num_ease_in_step: int = 300
    ease_exponent: float = 0.25

    frame_begin: Optional[int] = None
    frame_end: Optional[int] = None
    num_spins: float = 3.0
    num_frames_per_spin: int = 80

    def __setup__(self) -> None:
        assert self.ease_exponent > 0
        if self.up == 'disable':
            return
        up = (0, 1, 0) if self.up == '+y' else (0, 0, 1)
        self._cameras = Cameras.from_orbit(
            center=self.center,
            up=up,
            radius=self.radius,
            num_samples=self.spin_resolution,
            hfov_degree=self.hfov_degree,
            resolution=self.resolution,
            pitch_degree=self.pitch_degree,
        )
        self._sequence = {}

    def setup(self, num_steps: int) -> None:
        if self.up == 'disable':
            return
        # easing function: x**k / k
        # acc function:
        #   (ease_in/k * (x/ease_in)**k) if x <= ease_in
        #   ease_in/k + (x-ease_in) if x > ease_in
        frame_end = num_steps if self.frame_end is None else self.frame_end
        offset = 0 if self.frame_begin is None else self.frame_begin
        spin_per_step = self.num_spins / (self.num_ease_in_step * (1 / self.ease_exponent - 1) + frame_end)
        last_frame = -1
        for curr_step in range(1 + offset, num_steps + offset + 1):
            if curr_step <= self.num_ease_in_step:
                eased_step = (
                    self.num_ease_in_step / self.ease_exponent *
                    ((curr_step - 1) / self.num_ease_in_step) ** self.ease_exponent
                )
            else:
                eased_step = self.num_ease_in_step / self.ease_exponent + (curr_step - self.num_ease_in_step)
            curr_frame = spin_per_step * eased_step * self.num_frames_per_spin
            if int(curr_frame) > last_frame:
                self._sequence[curr_step - offset] = round(((spin_per_step * eased_step) * self.spin_resolution))
                last_frame = int(curr_frame)

    def get_camera(self, curr_step: int) -> Optional[Cameras]:
        if self.up == 'disable' or curr_step not in self._sequence:
            return None
        return self._cameras[self._sequence[curr_step] % self.spin_resolution]
