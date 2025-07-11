import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='torch.nn.modules.lazy', lineno=180)

from .mlp import MLP  # noqa: E402
from .module import Module, ParameterModule  # noqa: E402

__all__ = ['Module', 'MLP', 'ParameterModule']
