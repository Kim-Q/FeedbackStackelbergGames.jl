"""
Feedback Stackelberg Games - Python 实现包
基于 Newton-style Primal-Dual Interior Point Method 求解多玩家非线性约束动态博弈
"""

from .game_structs import (
    ConstrainedLQGame,
    NonlinearGame,
    Strategy,
    Trajectory,
)
from .lq_approximation import LqApproximation
from .forward_simulation import ForwardSimulator
from .nw_pdip_lq_solver import NwPdipFbstLqSolver
from .nw_pdip_line_search import NwPdipLineSearch
from .pdip_solver import PDIPSolver

__all__ = [
    "ConstrainedLQGame",
    "NonlinearGame",
    "Strategy",
    "Trajectory",
    "LqApproximation",
    "ForwardSimulator",
    "NwPdipFbstLqSolver",
    "NwPdipLineSearch",
    "PDIPSolver",
]
