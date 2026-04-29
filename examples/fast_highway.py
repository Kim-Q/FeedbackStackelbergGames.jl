"""
fast_highway.py - Python 版本对应 Julia fast_highway.jl

优化版 Highway 场景
- 更长预测时域 (horizon=100)
- 多层外层迭代 (num_outer_iter=10)
- 复杂的道路几何约束（倾斜边界）
- 碰撞避免约束
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import sys
sys.path.insert(0, '/workspace/python')

from feedback_stackelberg.pdip_solver import PDIPSolver
from feedback_stackelberg.config import PDIPConfig


@dataclass
class FastHighwayParameters:
    """Fast Highway 参数配置"""
    horizon: int = 20  # 减少时域以加快测试
    dt: float = 0.02
    nx: int = 8
    nu: int = 4
    m: int = 2
    n_players: int = 2
    
    # 道路几何参数
    x_right_corner: float = 1.1
    y_right_corner: float = 2.0
    base_x: float = 0.7
    road_length: float = 4.0
    
    # 初始状态 [x1, y1, v1, theta1, x2, y2, v2, theta2]
    x0: np.ndarray = None
    
    def __post_init__(self):
        if self.x0 is None:
            self.x0 = np.array([
                0.9,  # x1
                1.2,  # y1
                1.2,  # v1
                0.0,  # theta1
                0.5,  # x2
                0.4,  # y2
                1.6,  # v2
                0.0   # theta2
            ])


class FastHighwayScenario:
    """Fast Highway 场景类"""
    
    def __init__(self, params: Optional[FastHighwayParameters] = None):
        self.params = params or FastHighwayParameters()
        self.nx = self.params.nx
        self.nu = self.params.nu
        self.horizon = self.params.horizon
        
    def initial_state(self) -> np.ndarray:
        return self.params.x0.copy()
    
    def _road_constraints(self, x: float, y: float) -> float:
        """道路边界约束"""
        if y < self.params.road_length:
            return (self.params.road_length - y - 
                    (self.params.road_length - self.params.y_right_corner) / 
                    (self.params.x_right_corner - self.params.base_x) * 
                    (x - self.params.base_x))
        else:
            return self.params.base_x - x
    
    def _collision_avoidance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """碰撞避免约束"""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) - 0.5
    
    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """非线性动力学（Euler 积分）"""
        dt = self.params.dt
        x1, y1, v1, theta1 = state[0], state[1], state[2], state[3]
        x2, y2, v2, theta2 = state[4], state[5], state[6], state[7]
        
        u1, u2 = control[0], control[1]  # Player 1: acceleration, steering
        u3, u4 = control[2], control[3]  # Player 2: acceleration, steering
        
        next_state = np.array([
            x1 + dt * v1 * np.sin(theta1),
            y1 + dt * v1 * np.cos(theta1),
            v1 + dt * u1,
            theta1 + dt * u2,
            x2 + dt * v2 * np.sin(theta2),
            y2 + dt * v2 * np.cos(theta2),
            v2 + dt * u3,
            theta2 + dt * u4
        ])
        return next_state
    
    def stage_cost(self, state: np.ndarray, control: np.ndarray) -> float:
        """阶段代价"""
        # Player 1 (leader)
        cost1 = (4.0 * (state[0] - 0.5)**2 + 
                 (state[2] - state[6])**2 + 
                 control[0]**2 + control[1]**2)
        
        # Player 2 (follower)
        cost2 = (2.0 * (state[4] - 0.5)**2 + 
                 0.0 * (state[6] - self.params.x0[6])**2 + 
                 2.0 * state[7]**2 + 
                 control[2]**2 + control[3]**2)
        
        return float(cost1 + cost2)
    
    def terminal_cost(self, state: np.ndarray) -> float:
        """终端代价"""
        cost1 = 4.0 * (state[0] - 0.5)**2 + (state[2] - state[6])**2
        cost2 = 2.0 * (state[4] - 0.5)**2 + 0.0 * (state[6] - self.params.x0[6])**2 + 2.0 * state[7]**2
        return float(cost1 + cost2)
    
    def inequality_constraints(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """不等式约束：7 个约束"""
        x1, y1 = state[0], state[1]
        x2, y2 = state[4], state[5]
        
        constraints = np.array([
            x1 - 0.25,                                    # x1 >= 0.25
            x2 - 0.25,                                    # x2 >= 0.25
            self.params.x_right_corner - x1,              # x1 <= x_right_corner
            self.params.x_right_corner - x2,              # x2 <= x_right_corner
            self._road_constraints(x1, y1),               # Player 1 在道路内
            self._road_constraints(x2, y2),               # Player 2 在道路内
            self._collision_avoidance(x1, y1, x2, y2)     # 碰撞避免
        ])
        return constraints
    
    def terminal_inequality_constraints(self, state: np.ndarray) -> np.ndarray:
        """终端不等式约束"""
        return self.inequality_constraints(state, np.zeros(self.nu))
    
    def forward_simulation(self, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = np.zeros((self.horizon + 1, self.nx))
        states[0] = x0
        for t in range(self.horizon):
            states[t + 1] = self.dynamics(states[t], controls[t])
        return states
    
    def decision_shape(self) -> Tuple[int, ...]:
        return (self.horizon, self.nu)
    
    def rollout(self, x0: np.ndarray, decision: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        controls = decision.reshape(self.horizon, self.nu)
        states = self.forward_simulation(x0, controls)
        return states, controls
    
    def total_cost(self, states: np.ndarray, controls: np.ndarray) -> float:
        total = 0.0
        for t in range(self.horizon):
            total += self.stage_cost(states[t], controls[t])
        total += self.terminal_cost(states[-1])
        return total
    
    def collect_constraints(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        constraints = []
        for t in range(self.horizon):
            c = self.inequality_constraints(states[t], controls[t])
            constraints.append(c)
        c_terminal = self.terminal_inequality_constraints(states[-1])
        constraints.append(c_terminal)
        return np.concatenate(constraints)
    
    def format_iteration(self, decision: np.ndarray, total_cost: float, 
                        residual_norm: float, outer_iter: int, iteration: int) -> str:
        return (f"[outer {outer_iter+1}, iter {iteration+1}] "
                f"cost={total_cost:.6f}, residual={residual_norm:.3e}")


def main():
    print("=" * 60)
    print("Fast Highway - 优化版 Highway 场景")
    print("=" * 60)
    
    params = FastHighwayParameters()
    scenario = FastHighwayScenario(params)
    
    # 使用多层外层迭代（简化版用于测试）
    solver_config = PDIPConfig(
        max_iter=20,
        outer_iter=3,  # 减少外层迭代次数以加快测试
        barrier_mu=0.25,
        barrier_decay=0.25,  # σ = 4.0, decay = 1/σ
        residual_tol=1e-6,
        hessian_damping=1.0,
        print_iterations=True
    )
    
    solver = PDIPSolver(solver_config)
    result = solver.solve(scenario, scenario.initial_state())
    
    print("\n" + "=" * 60)
    print("求解完成!")
    print(f"最终损失：{result.loss_history[-1][-1]:.6f}")
    print(f"最终残差：{result.residual_history[-1][-1]:.3e}")
    print(f"外层迭代次数：{len(result.loss_history)}")
    print("=" * 60)
    
    # 显示部分轨迹
    print("\nPlayer 1 前 5 步位置:")
    print(result.states[:5, 0:2])
    
    print("\nPlayer 2 前 5 步位置:")
    print(result.states[:5, 4:6])
    
    return result


if __name__ == "__main__":
    main()
