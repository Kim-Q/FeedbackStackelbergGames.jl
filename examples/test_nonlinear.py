"""
test_nonlinear.py - Python 版本对应 Julia test_nonlinear.jl

一般非线性博弈测试
- 4 维状态，LTI 动力学（与 test_pdip 相同）
- 使用通用 game 类型定义非线性问题
- 无不等式约束（Gx, Gu, g 全为零）
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

import sys
sys.path.insert(0, '/workspace/python')

from feedback_stackelberg.pdip_solver import PDIPSolver
from feedback_stackelberg.config import PDIPConfig


@dataclass
class TestNonlinearParameters:
    """Test Nonlinear 参数配置"""
    horizon: int = 10
    nx: int = 4
    nu: int = 4
    m: int = 2
    n_players: int = 2
    
    A: np.ndarray = None
    B: np.ndarray = None
    Q1: np.ndarray = None
    Q2: np.ndarray = None
    R1: np.ndarray = None
    R2: np.ndarray = None
    x0: np.ndarray = None
    
    def __post_init__(self):
        if self.A is None:
            self.A = np.eye(self.nx)
        if self.B is None:
            self.B = 0.1 * np.eye(self.nx)
        if self.Q1 is None:
            self.Q1 = 4.0 * np.eye(self.nx)
        if self.Q2 is None:
            self.Q2 = 4.0 * np.eye(self.nx)
        if self.R1 is None:
            self.R1 = 2.0 * np.block([
                [np.eye(self.m), np.zeros((self.m, self.m))],
                [np.zeros((self.m, self.m)), np.zeros((self.m, self.m))]
            ])
        if self.R2 is None:
            self.R2 = 2.0 * np.block([
                [np.zeros((self.m, self.m)), np.zeros((self.m, self.m))],
                [np.zeros((self.m, self.m)), np.eye(self.m)]
            ])
        if self.x0 is None:
            self.x0 = np.array([1.0, 3.0, 2.0, 2.0])


class TestNonlinearScenario:
    """Test Nonlinear 场景类"""
    
    def __init__(self, params: Optional[TestNonlinearParameters] = None):
        self.params = params or TestNonlinearParameters()
        self.nx = self.params.nx
        self.nu = self.params.nu
        self.horizon = self.params.horizon
        
    def initial_state(self) -> np.ndarray:
        return self.params.x0.copy()
    
    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """LTI 动力学：x_{t+1} = A*x_t + B*u_t"""
        return self.params.A @ state + self.params.B @ control
    
    def stage_cost(self, state: np.ndarray, control: np.ndarray) -> float:
        """阶段代价：J1 + J2"""
        cost1 = 0.5 * (state @ self.params.Q1 @ state + 
                       control @ self.params.R1 @ control)
        cost2 = 0.5 * (state @ self.params.Q2 @ state + 
                       control @ self.params.R2 @ control)
        return float(cost1 + cost2)
    
    def terminal_cost(self, state: np.ndarray) -> float:
        """终端代价"""
        cost1 = 0.5 * state @ self.params.Q1 @ state
        cost2 = 0.5 * state @ self.params.Q2 @ state
        return float(cost1 + cost2)
    
    def inequality_constraints(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """无不等式约束"""
        return np.empty((0,), dtype=float)
    
    def terminal_inequality_constraints(self, state: np.ndarray) -> np.ndarray:
        """无终端不等式约束"""
        return np.empty((0,), dtype=float)
    
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
        return np.empty((0,), dtype=float)
    
    def format_iteration(self, decision: np.ndarray, total_cost: float, 
                        residual_norm: float, outer_iter: int, iteration: int) -> str:
        return (f"[outer {outer_iter+1}, iter {iteration+1}] "
                f"cost={total_cost:.6f}, residual={residual_norm:.3e}")


def main():
    print("=" * 60)
    print("Test Nonlinear - 一般非线性博弈测试")
    print("=" * 60)
    
    params = TestNonlinearParameters()
    scenario = TestNonlinearScenario(params)
    
    solver_config = PDIPConfig(
        max_iter=40,
        outer_iter=1,
        barrier_mu=1.0,
        barrier_decay=0.1,
        residual_tol=1e-8,
        hessian_damping=1.0,
        print_iterations=True
    )
    
    solver = PDIPSolver(solver_config)
    result = solver.solve(scenario, scenario.initial_state())
    
    print("\n" + "=" * 60)
    print("求解完成!")
    print(f"最终损失：{result.loss_history[-1][-1]:.6f}")
    print(f"最终残差：{result.residual_history[-1][-1]:.3e}")
    print("=" * 60)
    
    print("\n前 5 步状态:")
    print(result.states[:5])
    
    return result


if __name__ == "__main__":
    main()
