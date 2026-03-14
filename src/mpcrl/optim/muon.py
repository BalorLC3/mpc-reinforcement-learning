from typing import Literal, Optional, Union

import casadi as ca
import numpy as np
import numpy.typing as npt

from ..core.parameters import LearnableParametersDict
from ..core.schedulers import Scheduler
from .gradient_based_optimizer import GradientBasedOptimizer, LrType


class Muon(GradientBasedOptimizer[LrType]):
    r"""Despite of a sophisticated orthogonalization step Muon is
    indeed a first-order gradient based optimizer, based on :cite:``"""