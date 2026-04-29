from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class LQRParameters:
    """LQR 参数配置 - 支持标量和矩阵形式"""
    horizon: int = 30
    dt: float = 0.01  # 积分时间步长
    
    # 系统动态参数 (标量或矩阵)
    A: Union[float, np.ndarray] = 0.8181      # 系统矩阵 (标量)
    B1: Union[float, np.ndarray] = 0.8175     # 玩家 1 控制输入矩阵
    B2: Union[float, np.ndarray] = -0.7224    # 玩家 2 控制输入矩阵
    
    # 玩家 1 的代价函数参数
    Q1: Union[float, np.ndarray] = 0.1499     # 状态权重
    Theta1: Union[float, np.ndarray] = 0.3245 # 对 u2 的交叉权重 (Θ₁)
    R11: Union[float, np.ndarray] = 0.5186    # 对 u1 的权重
    R12: Union[float, np.ndarray] = 0         # 对 u2 的权重
    
    # 玩家 2 的代价函数参数
    Q2: Union[float, np.ndarray] = 0.6596     # 状态权重
    Theta2: Union[float, np.ndarray] = 0.4002 # 对 u1 的交叉权重 (Θ₂)
    R22: Union[float, np.ndarray] = 0.9730    # 对 u2 的权重
    R21: Union[float, np.ndarray] = 0         # 对 u1 的权重
    
    # FSE (Feedback Stackelberg Equilibrium) 解（运行后可更新为收敛增益）
    FSE_solutions: Dict[str, Dict[str, Union[float, list[float]]]] = field(
        default_factory=lambda: {
            "SE 1": {"K1_star": -0.6662, "K2_star": 1.1621},
            "SE 2": {"K1_star": -0.9542, "K2_star": 0.8523},
            "SE 3": {"K1_star": -1.6526, "K2_star": 0.7539}
        }
    )
    
    # 终端代价权重 (可选，默认与阶段代价相同)
    Q_terminal_leader: Optional[Union[float, np.ndarray]] = None
    Q_terminal_follower: Optional[Union[float, np.ndarray]] = None
    
    # 初始状态
    x0: Union[float, np.ndarray] = 1.0
    
    # 非线性动态函数 (可选)
    f_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    g_leader_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    g_follower_func: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self) -> None:
        self.normalize()

    def normalize(self) -> None:
        """将所有参数转换为 numpy 数组"""
        # 系统动态参数
        if isinstance(self.A, (int, float)):
            self.A = np.array([[self.A]], dtype=float)
        else:
            self.A = np.asarray(self.A, dtype=float)
        
        if isinstance(self.B1, (int, float)):
            self.B1 = np.array([[self.B1]], dtype=float)
        else:
            self.B1 = np.asarray(self.B1, dtype=float)
            
        if isinstance(self.B2, (int, float)):
            self.B2 = np.array([[self.B2]], dtype=float)
        else:
            self.B2 = np.asarray(self.B2, dtype=float)
        
        # 代价函数参数 - 玩家 1 (leader)
        if isinstance(self.Q1, (int, float)):
            self.Q1 = np.array([[self.Q1]], dtype=float)
        else:
            self.Q1 = np.asarray(self.Q1, dtype=float)
            
        if isinstance(self.Theta1, (int, float)):
            self.Theta1 = np.array([[self.Theta1]], dtype=float)
        else:
            self.Theta1 = np.asarray(self.Theta1, dtype=float)
            
        if isinstance(self.R11, (int, float)):
            self.R11 = np.array([[self.R11]], dtype=float)
        else:
            self.R11 = np.asarray(self.R11, dtype=float)
            
        if isinstance(self.R12, (int, float)):
            self.R12 = np.array([[self.R12]], dtype=float)
        else:
            self.R12 = np.asarray(self.R12, dtype=float)
        
        # 代价函数参数 - 玩家 2 (follower)
        if isinstance(self.Q2, (int, float)):
            self.Q2 = np.array([[self.Q2]], dtype=float)
        else:
            self.Q2 = np.asarray(self.Q2, dtype=float)
            
        if isinstance(self.Theta2, (int, float)):
            self.Theta2 = np.array([[self.Theta2]], dtype=float)
        else:
            self.Theta2 = np.asarray(self.Theta2, dtype=float)
            
        if isinstance(self.R22, (int, float)):
            self.R22 = np.array([[self.R22]], dtype=float)
        else:
            self.R22 = np.asarray(self.R22, dtype=float)
            
        if isinstance(self.R21, (int, float)):
            self.R21 = np.array([[self.R21]], dtype=float)
        else:
            self.R21 = np.asarray(self.R21, dtype=float)
        
        # 初始状态
        if isinstance(self.x0, (int, float)):
            self.x0 = np.array([self.x0], dtype=float)
        else:
            self.x0 = np.asarray(self.x0, dtype=float).reshape(-1)
        
        # 终端代价
        if self.Q_terminal_leader is None:
            self.Q_terminal_leader = self.Q1.copy()
        elif isinstance(self.Q_terminal_leader, (int, float)):
            self.Q_terminal_leader = np.array([[self.Q_terminal_leader]], dtype=float)
        else:
            self.Q_terminal_leader = np.asarray(self.Q_terminal_leader, dtype=float)
            
        if self.Q_terminal_follower is None:
            self.Q_terminal_follower = self.Q2.copy()
        elif isinstance(self.Q_terminal_follower, (int, float)):
            self.Q_terminal_follower = np.array([[self.Q_terminal_follower]], dtype=float)
        else:
            self.Q_terminal_follower = np.asarray(self.Q_terminal_follower, dtype=float)

    def apply_overrides(self, overrides: Dict[str, Any]) -> "LQRParameters":
        for key, value in overrides.items():
            setattr(self, key, value)
        self.normalize()
        return self


class LQRScenario:
    """LQR 场景类 - 实现标量/矩阵 LQR 动态和代价函数"""
    
    def __init__(self, params: Optional[LQRParameters] = None):
        self.params = params or LQRParameters()
        self.nx = int(self.params.A.shape[0])
        self._nu_leader = int(self.params.B1.shape[1])
        self._nu_follower = int(self.params.B2.shape[1])
        self.nu = self._nu_leader + self._nu_follower
        self.n_players = 2
        self.players_u_index_list = (
            tuple(range(self._nu_leader)),
            tuple(range(self._nu_leader, self.nu)),
        )

    def initial_state(self) -> np.ndarray:
        return self.params.x0.copy()

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        状态动态方程：dot{x} = A*x + B1*u1 + B2*u2
        使用 RK45 数值积分
        """
        u_leader, u_follower = self._split_controls(control)
        return self._integrate_dynamics(state, u_leader, u_follower)

    def stage_cost(self, state: np.ndarray, control: np.ndarray) -> float:
        """
        阶段代价函数：
        J1 = 0.5 * [Q1*x^2 + Theta1*u2^2 + R11*u1^2 + R12*u2^2]
        J2 = 0.5 * [Q2*x^2 + Theta2*u1^2 + R22*u2^2 + R21*u1^2]
        返回总代价 J1 + J2
        """
        u_leader, u_follower = self._split_controls(control)
        
        # 玩家 1 (leader) 的代价
        leader_cost = self._player1_cost(state, u_leader, u_follower)
        
        # 玩家 2 (follower) 的代价
        follower_cost = self._player2_cost(state, u_leader, u_follower)
        
        return float(leader_cost + follower_cost)

    def terminal_cost(self, state: np.ndarray) -> float:
        """终端代价"""
        leader_cost = 0.5 * float(state @ self.params.Q_terminal_leader @ state)
        follower_cost = 0.5 * float(state @ self.params.Q_terminal_follower @ state)
        return float(leader_cost + follower_cost)

    def inequality_constraints(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return np.empty((0,), dtype=float)

    def terminal_inequality_constraints(self, state: np.ndarray) -> np.ndarray:
        return np.empty((0,), dtype=float)

    def forward_simulation(self, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """前向模拟得到状态轨迹"""
        states = np.zeros((self.params.horizon + 1, self.nx), dtype=float)
        states[0] = x0
        for t in range(self.params.horizon):
            states[t + 1] = self.dynamics(states[t], controls[t])
        return states

    def decision_shape(self) -> tuple[int, int]:
        """决策变量形状（反馈增益 K，满足 u = K x）"""
        return (self.nu, self.nx)

    def rollout(self, x0: np.ndarray, decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """根据反馈增益 K 生成状态和控制序列"""
        gains = self._reshape_gains(decision)
        states = np.zeros((self.params.horizon + 1, self.nx), dtype=float)
        controls = np.zeros((self.params.horizon, self.nu), dtype=float)
        states[0] = x0
        for t in range(self.params.horizon):
            controls[t] = (gains @ states[t].reshape(-1, 1)).ravel()
            states[t + 1] = self.dynamics(states[t], controls[t])
        return states, controls

    def extract_feedback_gains(self, decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gains = self._reshape_gains(decision)
        k1 = gains[: self._nu_leader, :]
        k2 = gains[self._nu_leader :, :]
        return k1, k2

    def format_iteration(
        self,
        decision: np.ndarray,
        total_cost: float,
        residual_norm: float,
        outer_iter: int,
        iteration: int,
    ) -> str:
        k1, k2 = self.extract_feedback_gains(decision)
        k1_str = np.array2string(k1, precision=4, separator=",", suppress_small=True)
        k2_str = np.array2string(k2, precision=4, separator=",", suppress_small=True)
        return (
            f"[outer {outer_iter + 1}, iter {iteration + 1}] "
            f"cost={total_cost:.6f}, residual={residual_norm:.3e}, "
            f"K1={k1_str}, K2={k2_str}"
        )

    def update_fse_reference(self, decision: np.ndarray) -> None:
        k1, k2 = self.extract_feedback_gains(decision)
        self.params.FSE_solutions = {
            "converged": {
                "K1_star": self._serialize_gain(k1),
                "K2_star": self._serialize_gain(k2),
            }
        }

    def serialize_feedback_gains(self, decision: np.ndarray) -> Dict[str, Union[float, list[float]]]:
        k1, k2 = self.extract_feedback_gains(decision)
        return {
            "K1": self._serialize_gain(k1),
            "K2": self._serialize_gain(k2),
        }

    def total_cost(self, states: np.ndarray, controls: np.ndarray) -> float:
        """计算总代价"""
        total = 0.0
        for t in range(self.params.horizon):
            total += self.stage_cost(states[t], controls[t])
        total += self.terminal_cost(states[-1])
        return float(total)

    def collect_constraints(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        return np.empty((0,), dtype=float)

    def _split_controls(self, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        control = np.asarray(control, dtype=float).reshape(-1)
        leader = control[: self._nu_leader]
        follower = control[self._nu_leader :]
        return leader, follower

    def _integrate_dynamics(
        self, state: np.ndarray, u_leader: np.ndarray, u_follower: np.ndarray
    ) -> np.ndarray:
        """
        使用 scipy.integrate.solve_ivp 进行 RK45 积分
        dot{x} = A*x + B1*u1 + B2*u2
        
        注意：对于标量情况，需要将控制量转换为适当的形状
        """
        # 确保状态是列向量 (nx, 1)
        x = state.reshape(-1, 1) if state.ndim == 1 else state
        
        # 将控制量转为列向量
        u1_val = np.asarray(u_leader).reshape(-1, 1) if u_leader.size > 0 else np.zeros((self._nu_leader, 1))
        u2_val = np.asarray(u_follower).reshape(-1, 1) if u_follower.size > 0 else np.zeros((self._nu_follower, 1))
        
        # 定义微分方程
        def dxdt(t, x_flat):
            x_vec = x_flat.reshape(-1, 1)
            return (self.params.A @ x_vec + self.params.B1 @ u1_val + self.params.B2 @ u2_val).ravel()
        
        # 积分
        sol = solve_ivp(dxdt, (0, self.params.dt), state.ravel(), method='RK45', rtol=1e-8, atol=1e-10)
        return sol.y[:, -1]

    def _player1_cost(self, state: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> float:
        """
        玩家 1 (leader) 的代价函数：
        J1 = 0.5 * [x'*Q1*x + u2'*Theta1*u2 + u1'*R11*u1 + u2'*R12*u2]
        """
        state_term = float(state @ self.params.Q1 @ state)
        u1_term = float(u1 @ self.params.R11 @ u1)
        u2_theta_term = float(u2 @ self.params.Theta1 @ u2)
        u2_r12_term = float(u2 @ self.params.R12 @ u2)
        return 0.5 * (state_term + u1_term + u2_theta_term + u2_r12_term)

    def _player2_cost(self, state: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> float:
        """
        玩家 2 (follower) 的代价函数：
        J2 = 0.5 * [x'*Q2*x + u1'*Theta2*u1 + u2'*R22*u2 + u1'*R21*u1]
        """
        state_term = float(state @ self.params.Q2 @ state)
        u2_term = float(u2 @ self.params.R22 @ u2)
        u1_theta_term = float(u1 @ self.params.Theta2 @ u1)
        u1_r21_term = float(u1 @ self.params.R21 @ u1)
        return 0.5 * (state_term + u2_term + u1_theta_term + u1_r21_term)

    def _reshape_gains(self, decision: np.ndarray) -> np.ndarray:
        return np.asarray(decision, dtype=float).reshape(self.nu, self.nx)

    def _serialize_gain(self, gain: np.ndarray) -> Union[float, list[float]]:
        gain_array = np.asarray(gain, dtype=float)
        if gain_array.size == 1:
            return float(gain_array.ravel()[0])
        return gain_array.tolist()


def load_lqr_overrides(path: Union[str, Path]) -> Dict[str, Any]:
    """加载 LQR 参数覆盖"""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return _load_overrides_from_npz(data)
    if path.suffix == ".json":
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        return _load_overrides_from_json(raw)
    raise ValueError(f"Unsupported LQR parameter file: {path}")


def _load_overrides_from_npz(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key in data.files:
        value = data[key]
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if key == "dynamics":
            overrides[key] = str(value)
        else:
            overrides[key] = value
    return overrides


def _load_overrides_from_json(raw: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, value in raw.items():
        if key == "dynamics":
            overrides[key] = str(value)
        elif isinstance(value, list):
            overrides[key] = np.asarray(value, dtype=float)
        else:
            overrides[key] = value
    return overrides
