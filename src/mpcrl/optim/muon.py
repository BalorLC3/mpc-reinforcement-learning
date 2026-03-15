from typing import Literal, Optional, Union

import casadi as ca
import numpy as np
import numpy.typing as npt

from ..core.parameters import LearnableParametersDict
from ..core.schedulers import Scheduler
from .gradient_based_optimizer import GradientBasedOptimizer, LrType


class Muon(GradientBasedOptimizer[LrType]):
    r"""Despite of a sophisticated orthogonalization step Muon is indeed a
    first-order gradient based optimizer, based on :cite: ``"""

    def __init__(
            self,
            learning_rate: Union[LrType, Scheduler[LrType]],

    ) -> None:
        super().__init__(learning_rate)
        ...


def _muon(
        step: int,
        g: np.ndarray,
):
    ...