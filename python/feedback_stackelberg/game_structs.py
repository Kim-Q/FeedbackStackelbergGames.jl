"""
game_structs.py
==============
博弈数据结构定义模块

包含以下数据类：
  - ConstrainedLQGame  : 受约束的线性二次型博弈参数（LQ近似子问题）
  - NonlinearGame      : 一般非线性博弈参数（原始问题）
  - Strategy           : 反馈策略 u_t = -P_t * x_t - alpha_t
  - Trajectory         : 博弈轨迹（状态、控制、所有拉格朗日乘子）
"""

import copy
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np


# ---------------------------------------------------------------------------
# ConstrainedLQGame
# ---------------------------------------------------------------------------

@dataclass
class ConstrainedLQGame:
    """
    受约束的线性二次型博弈数据结构

    存储每个时间步的动力学矩阵、代价矩阵以及等式/不等式约束矩阵。
    通常由 LqApproximation 对非线性博弈在某操作点处线性化后生成。

    约定（与 Julia 原版一致）：
      动力学     : x_{t+1} = A_t * x_t + B_t * u_t + c_t
      代价函数   : J_i = 0.5*(x'Q_i x + u'R_i u + 2u'S_i x + 2q_i'x + 2r_i'u)
      等式约束   : H_i(x,u) := Hx_i*x + Hu_i*u + h_i = 0
      不等式约束 : G_i(x,u) := Gx_i*x + Gu_i*u + g_i >= 0

    Notes
    -----
    索引约定：Python 使用 0-based 索引。
    players_u_index_list[i] 是一个整数列表，表示玩家 i 控制的 u 分量索引（0-based）。
    例：nu=4, 两玩家各控制 2 个分量 → [[0,1],[2,3]]
    """

    # ------------------------------------------------------------------
    # 基本维度
    # ------------------------------------------------------------------
    horizon: int                           # 预测时域 T（步数）
    n_players: int                         # 玩家数量（当前仅支持 2 人）
    nx: int                                # 状态维度
    nu: int                                # 控制总维度
    players_u_index_list: List[List[int]]  # 各玩家控制输入的索引（0-based）

    # ------------------------------------------------------------------
    # 动力学矩阵列表（长度均为 T）
    # ------------------------------------------------------------------
    A_list: List[np.ndarray]               # 状态转移矩阵 A_t ∈ R^{nx×nx}
    B_list: List[np.ndarray]               # 控制增益矩阵 B_t ∈ R^{nx×nu}
    c_list: List[np.ndarray]               # 线性化残差向量 c_t ∈ R^{nx}

    # ------------------------------------------------------------------
    # 代价矩阵列表（Q/q 长度为 T+1，S/R/r 长度为 T，第二维为 n_players）
    # ------------------------------------------------------------------
    Q_list: List[List[np.ndarray]]         # 状态代价矩阵 Q_t^i ∈ R^{nx×nx}
    S_list: List[List[np.ndarray]]         # 交叉代价矩阵 S_t^i ∈ R^{nu×nx}
    R_list: List[List[np.ndarray]]         # 控制代价矩阵 R_t^i ∈ R^{nu×nu}
    q_list: List[List[np.ndarray]]         # 状态线性代价向量 q_t^i ∈ R^{nx}
    r_list: List[List[np.ndarray]]         # 控制线性代价向量 r_t^i ∈ R^{nu}

    # ------------------------------------------------------------------
    # 等式约束（长度 T，终端另存）
    # ------------------------------------------------------------------
    equality_constraints_size: int = 1     # 等式约束维度 l
    Hx_list: List = field(default_factory=list)   # Hx_t^i ∈ R^{l×nx}
    Hu_list: List = field(default_factory=list)   # Hu_t^i ∈ R^{l×nu}
    h_list: List = field(default_factory=list)    # h_t^i ∈ R^{l}
    HxT: List = field(default_factory=list)       # 终端 HxT^i ∈ R^{l×nx}
    hxT: List = field(default_factory=list)       # 终端 hxT^i ∈ R^{l}

    # ------------------------------------------------------------------
    # 不等式约束（长度 T，终端另存）
    # 约定：G_i(x,u) >= 0，slack s 满足 G_i(x,u) = s >= 0
    # ------------------------------------------------------------------
    inequality_constraints_size: int = 1   # 不等式约束维度 ll
    Gx_list: List = field(default_factory=list)   # Gx_t^i ∈ R^{ll×nx}
    Gu_list: List = field(default_factory=list)   # Gu_t^i ∈ R^{ll×nu}
    g_list: List = field(default_factory=list)    # g_t^i ∈ R^{ll}
    GxT: List = field(default_factory=list)       # 终端 GxT^i ∈ R^{ll×nx}
    gxT: List = field(default_factory=list)       # 终端 gxT^i ∈ R^{ll}

    # ------------------------------------------------------------------
    # 初始状态
    # ------------------------------------------------------------------
    x0: np.ndarray = field(default_factory=lambda: np.array([]))

    def copy(self):
        """深拷贝此博弈对象"""
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# NonlinearGame
# ---------------------------------------------------------------------------

@dataclass
class NonlinearGame:
    """
    一般非线性博弈数据结构

    存储非线性动力学、代价函数和约束函数的函数定义（可调用对象）。
    用作 LqApproximation 的输入，以及在线搜索中计算 KKT 残差。

    约定：
      动力学     : x_{t+1} = f_t(x_t, u_t)
      等式约束   : h_t^i(x_t, u_t) = 0   （约束值应为 0）
      不等式约束 : g_t^i(x_t, u_t) >= 0  （约束值 ≥ 0 则满足）
    """

    # ------------------------------------------------------------------
    # 基本维度
    # ------------------------------------------------------------------
    horizon: int
    n_players: int
    nx: int
    nu: int
    players_u_index_list: List[List[int]]

    # ------------------------------------------------------------------
    # 函数定义
    # ------------------------------------------------------------------
    f_list: List[Callable]                         # f_t(x, u) -> x_next ∈ R^{nx}
    costs_list: List[List[Callable]]               # l_t^i(x, u) -> scalar
    terminal_costs_list: List[Callable]            # V^i(x) -> scalar
    x0: np.ndarray                                 # 初始状态

    equality_constraints_list: List[List[Callable]]       # h_t^i(z) -> R^{l},  z=[x;u]
    terminal_equality_constraints_list: List[Callable]    # h_T^i(x) -> R^{l}
    inequality_constraints_list: List[List[Callable]]     # g_t^i(z) -> R^{ll}, z=[x;u]
    terminal_inequality_constraints_list: List[Callable]  # g_T^i(x) -> R^{ll}

    equality_constraints_size: int = 1
    inequality_constraints_size: int = 1


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """
    线性反馈策略数据结构

    在每个时刻 t，控制律为：
        u_t = -P_t * x_t - alpha_t

    其中 P_t 是状态反馈增益矩阵，alpha_t 是前馈偏置向量。
    """

    P: List[np.ndarray]      # 反馈增益矩阵列表，P[t] ∈ R^{nu×nx}，长度为 T
    alpha: List[np.ndarray]  # 前馈偏置向量列表，alpha[t] ∈ R^{nu}，长度为 T


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """
    博弈轨迹数据结构

    包含状态序列、控制序列以及所有拉格朗日乘子。
    KKT 最优性条件中的对偶变量均存储于此。

    长度约定（T = horizon）：
      x     : T+1（包含 x_0 和 x_T）
      u     : T
      lam   : T（动力学拉格朗日乘子 λ^i，合并为 nx*n_players 向量）
      eta   : T-1（策略一致性乘子 η，大小 nu）
      psi   : T（追随者策略乘子 ψ，大小 m = nu/n_players）
      mu    : T+1（等式约束乘子 μ，大小 l*n_players）
      gamma : T+1（不等式约束乘子 γ，大小 ll*n_players）
      s     : T+1（不等式约束松弛变量，大小 ll*n_players，s > 0）
    """

    x: List[np.ndarray]      # 状态序列
    u: List[np.ndarray]      # 控制序列
    lam: List[np.ndarray]    # 动力学乘子 λ
    eta: List[np.ndarray]    # 策略一致性乘子 η
    psi: List[np.ndarray]    # 追随者策略乘子 ψ

    mu: List[np.ndarray] = field(default_factory=list)    # 等式约束乘子 μ
    gamma: List[np.ndarray] = field(default_factory=list) # 不等式约束乘子 γ
    s: List[np.ndarray] = field(default_factory=list)     # 松弛变量 s > 0

    def copy(self):
        """深拷贝此轨迹对象"""
        return copy.deepcopy(self)
