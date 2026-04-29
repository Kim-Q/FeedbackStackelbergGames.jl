"""
test_pdip.py - Python 版本对应 Julia test_pdip.jl

线性二次 Stackelberg 博弈基础测试
- 4 维状态，LTI 动力学
- 2 个玩家，每个玩家 2 维控制
- 简单的等式和不等式约束（全零，实际无约束）
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Any, Optional

# 添加路径以导入 feedback_stackelberg 模块
import sys
sys.path.insert(0, '/workspace/python')

from feedback_stackelberg.pdip_solver import PDIPSolver, PDIPResult
from feedback_stackelberg.config import PDIPConfig


@dataclass
class TestPDIPParameters:
    """Test PDIP 参数配置"""
    horizon: int = 10
    nx: int = 4  # 状态维度
    nu: int = 4  # 总控制维度
    m: int = 2   # 每个玩家的控制维度
    n_players: int = 2
    
    # 系统动态矩阵
    A: np.ndarray = None
    B: np.ndarray = None
    
    # 代价函数权重
    Q1: np.ndarray = None
    Q2: np.ndarray = None
    R1: np.ndarray = None
    R2: np.ndarray = None
    
    # 初始状态
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


class TestPDIPScenario:
    """Test PDIP 场景类"""
    
    def __init__(self, params: Optional[TestPDIPParameters] = None):
        self.params = params or TestPDIPParameters()
        self.nx = self.params.nx
        self.nu = self.params.nu
        self.n_players = self.params.n_players
        self.horizon = self.params.horizon
        self.players_u_index_list = [range(0, 2), range(2, 4)]
        
    def initial_state(self) -> np.ndarray:
        return self.params.x0.copy()
    
    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """LTI 动力学：x_{t+1} = A*x_t + B*u_t"""
        return self.params.A @ state + self.params.B @ control
    
    def stage_cost(self, state: np.ndarray, control: np.ndarray) -> float:
        """阶段代价：J1 + J2"""
        u1 = control[0:2]
        u2 = control[2:4]
        
        # 玩家 1 的代价
        cost1 = 0.5 * (state @ self.params.Q1 @ state + 
                       control @ self.params.R1 @ control)
        
        # 玩家 2 的代价
        cost2 = 0.5 * (state @ self.params.Q2 @ state + 
                       control @ self.params.R2 @ control)
        
        return float(cost1 + cost2)
    
    def terminal_cost(self, state: np.ndarray) -> float:
        """终端代价"""
        cost1 = 0.5 * state @ self.params.Q1 @ state
        cost2 = 0.5 * state @ self.params.Q2 @ state
        return float(cost1 + cost2)
    
    def inequality_constraints(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """不等式约束（空约束，用于测试）"""
        # Gx1_terminal = [0 0 0 1.0], g1_terminal = [-1.0]
        # 即：x[3] >= 1.0
        con1 = state[3] - 1.0
        con2 = state[3] - 1.0
        return np.array([con1, con2])
    
    def terminal_inequality_constraints(self, state: np.ndarray) -> np.ndarray:
        """终端不等式约束"""
        return self.inequality_constraints(state, np.zeros(self.nu))
    
    def forward_simulation(self, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """前向模拟"""
        states = np.zeros((self.horizon + 1, self.nx))
        states[0] = x0
        for t in range(self.horizon):
            states[t + 1] = self.dynamics(states[t], controls[t])
        return states
    
    def decision_shape(self) -> Tuple[int, ...]:
        """决策变量形状"""
        return (self.horizon, self.nu)
    
    def rollout(self, x0: np.ndarray, decision: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """根据控制序列生成轨迹"""
        controls = decision.reshape(self.horizon, self.nu)
        states = self.forward_simulation(x0, controls)
        return states, controls
    
    def total_cost(self, states: np.ndarray, controls: np.ndarray) -> float:
        """计算总代价"""
        total = 0.0
        for t in range(self.horizon):
            total += self.stage_cost(states[t], controls[t])
        total += self.terminal_cost(states[-1])
        return total
    
    def collect_constraints(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """收集所有约束"""
        constraints = []
        for t in range(self.horizon):
            c = self.inequality_constraints(states[t], controls[t])
            constraints.append(c)
        # 终端约束
        c_terminal = self.terminal_inequality_constraints(states[-1])
        constraints.append(c_terminal)
        return np.concatenate(constraints)
    
    def format_iteration(self, decision: np.ndarray, total_cost: float, 
                        residual_norm: float, outer_iter: int, iteration: int) -> str:
        """格式化输出迭代信息"""
        return (f"[outer {outer_iter+1}, iter {iteration+1}] "
                f"cost={total_cost:.6f}, residual={residual_norm:.3e}")


def main():
    """主函数"""
    print("=" * 60)
    print("Test PDIP - 线性二次 Stackelberg 博弈基础测试")
    print("=" * 60)
    
    # 初始化参数和场景
    params = TestPDIPParameters()
    scenario = TestPDIPScenario(params)
    
    # 配置求解器
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
    
    # 求解
    result = solver.solve(scenario, scenario.initial_state())
    
    # 输出结果
    print("\n" + "=" * 60)
    print("求解完成!")
    print(f"最终损失：{result.loss_history[-1][-1]:.6f}")
    print(f"最终残差：{result.residual_history[-1][-1]:.3e}")
    print(f"状态轨迹形状：{result.states.shape}")
    print(f"控制轨迹形状：{result.controls.shape}")
    print("=" * 60)
    
    # 显示部分结果
    print("\n前 5 步状态:")
    print(result.states[:5])
    
    print("\n前 5 步控制:")
    print(result.controls[:5])
    
    return result


if __name__ == "__main__":
    main()
