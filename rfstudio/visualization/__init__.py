import logging

from ._base import Visualizer, vis_3dgs
from ._colmap import vis_colmap
from ._figure_highlighter import highlight
from ._tabular_figures import TabularFigures

# Suppress INFO logs from the websockets library
logging.getLogger("websockets").setLevel(logging.WARNING)

__all__ = [
    'vis_3dgs',
    'vis_colmap',
    'Visualizer',
    'highlight',
    'TabularFigures',
]
