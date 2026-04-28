"""
PDIP (Primal-Dual Interior Point Method) 求解器模块

实现牛顿对偶内点法求解带不等式约束的优化问题。
算法核心思想：
1. 使用障碍函数将不等式约束转化为目标函数的一部分
2. 通过牛顿法迭代求解 KKT 条件
3. 逐步减小障碍参数，逼近原问题的最优解
"""

from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import numpy as np

from feedback_stackelberg.config import PDIPConfig


@dataclass
class PDIPResult:
    """PDIP 算法结果数据类
    
    Attributes:
        states: 状态轨迹 [horizon+1, nx]
        controls: 控制输入序列 [horizon, nu]
        slack: 松弛变量，用于将不等式约束转为等式
        dual: 对偶变量（拉格朗日乘子）
        loss_history: 每次外层迭代的损失函数收敛历史
        residual_history: 每次外层迭代的 KKT 残差收敛历史
    """
    states: np.ndarray
    controls: np.ndarray
    slack: np.ndarray
    dual: np.ndarray
    loss_history: List[List[float]]
    residual_history: List[List[float]]


class PDIPSolver:
    """
    原始 - 对偶内点法 (Primal-Dual Interior Point Method) 求解器
    
    使用牛顿法求解带不等式约束的优化问题：
        min f(u)
        s.t. g(u) <= 0
    
    通过引入松弛变量 s > 0，将不等式转为等式：g(u) + s = 0
    构造障碍问题：min f(u) - μ * Σlog(s_i)
    """
    
    def __init__(self, config: Union[PDIPConfig, None] = None):
        """初始化求解器
        
        Args:
            config: PDIP 配置参数，若为 None 则使用默认配置
        """
        self.config = config or PDIPConfig()

    def solve(self, scenario: Any, initial_state: Union[np.ndarray, None] = None) -> PDIPResult:
        """
        求解 Stackelberg 博弈问题
        
        Args:
            scenario: 场景对象，提供动力学、成本函数和约束
            initial_state: 初始状态，若为 None 则使用场景的默认初始状态
            
        Returns:
            PDIPResult: 包含状态轨迹、控制输入、松弛变量、对偶变量及收敛历史
        """
        # ========== 步骤 1: 初始化原始变量和对偶变量 ==========
        # 获取初始状态
        x0 = initial_state.copy() if initial_state is not None else scenario.initial_state()
        
        # 初始化控制输入为零
        controls = np.zeros((scenario.params.horizon, scenario.nu), dtype=float)
        
        # 获取约束维度并初始化松弛变量和对偶变量
        # 松弛变量 s > 0，对偶变量 λ > 0
        constraint_dim = scenario.collect_constraints(
            scenario.forward_simulation(x0, controls), controls
        ).shape[0]
        slack = np.ones(constraint_dim, dtype=float) * self.config.initial_slack
        dual = np.ones(constraint_dim, dtype=float) * self.config.initial_dual
        
        # 记录收敛历史
        loss_history: List[List[float]] = []
        residual_history: List[List[float]] = []
        
        # 障碍参数 μ，控制对数障碍的权重
        mu = self.config.barrier_mu

        # ========== 步骤 2: 外层循环 - 更新障碍参数 ==========
        for outer_iter in range(self.config.outer_iter):
            inner_loss: List[float] = []
            inner_residual: List[float] = []
            
            # ========== 步骤 3: 内层牛顿迭代 ==========
            for iteration in range(self.config.max_iter):
                # ----- 步骤 3.1: 前向模拟得到状态轨迹 -----
                # 根据当前控制输入和初始状态，正向传播得到状态序列
                states = scenario.forward_simulation(x0, controls)
                
                # ----- 步骤 3.2: 计算约束与目标函数 -----
                # 收集所有时刻的不等式约束 g(u) <= 0
                constraints = scenario.collect_constraints(states, controls)
                # 计算总成本函数 f(u)
                total_cost = scenario.total_cost(states, controls)
                
                # ----- 步骤 3.3: 构造 KKT 残差 -----
                # KKT 条件包括：
                # 1. 平稳性条件：∇f(u) - J(u)^T * λ = 0
                # 2. 原始可行性：g(u) + s = 0
                # 3. 互补松弛条件：s_i * λ_i = μ
                
                # 使用有限差分计算梯度 ∇f(u)
                grad = self._finite_difference_gradient(scenario, x0, controls)
                
                # 使用有限差分计算约束雅可比矩阵 J(u) = ∂g/∂u
                jacobian = self._finite_difference_jacobian(scenario, x0, controls, constraints)
                
                # 计算三类残差
                r_dual, r_pri, r_cent = self._kkt_residuals(
                    grad, jacobian, constraints, slack, dual, mu
                )
                
                # 计算残差范数作为收敛判据
                residual_norm = self._residual_norm(r_dual, r_pri, r_cent)
                
                inner_loss.append(total_cost)
                inner_residual.append(residual_norm)
                
                # 检查收敛性：若残差小于阈值则停止迭代
                if residual_norm < self.config.residual_tol:
                    break
                
                # ----- 步骤 3.4: 求解牛顿方向 -----
                # 线性化 KKT 条件，得到如下线性系统：
                # [ H   -J^T   0  ] [Δu  ]   [r_dual]
                # [ J    0    -I  ] [Δλ  ] = [-r_pri ]
                # [ 0    S    Λ  ] [Δs  ]   [-r_cent]
                # 其中 H 为 Hessian 近似（这里用阻尼单位阵），S=diag(s), Λ=diag(λ)
                delta_u, delta_dual, delta_slack = self._solve_kkt_system(
                    jacobian, r_dual, r_pri, r_cent, slack, dual
                )
                
                # ----- 步骤 3.5: 线搜索保证正性约束 -----
                # 由于要求 s > 0 和 λ > 0，需要限制步长
                step = self._line_search(slack, dual, delta_slack, delta_dual)
                
                # ----- 步骤 3.6: 更新原始变量与对偶变量 -----
                controls = self._update_controls(controls, delta_u, step)
                slack = slack + step * delta_slack
                dual = dual + step * delta_dual
                
            loss_history.append(inner_loss)
            residual_history.append(inner_residual)
            
            # 衰减障碍参数，逐步逼近原问题的解
            # μ → 0 时，障碍问题的解趋向于原问题的解
            mu *= self.config.barrier_decay

        # 最终前向传播获取最优状态轨迹
        states = scenario.forward_simulation(x0, controls)
        return PDIPResult(
            states=states,
            controls=controls,
            slack=slack,
            dual=dual,
            loss_history=loss_history,
            residual_history=residual_history,
        )

    def _finite_difference_gradient(
        self, scenario: Any, x0: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        """
        使用有限差分法计算目标函数关于控制输入的梯度
        
        ∂f/∂u_i ≈ (f(u + ε*e_i) - f(u)) / ε
        
        Args:
            scenario: 场景对象
            x0: 初始状态
            controls: 当前控制输入
            
        Returns:
            梯度向量 [nu*horizon]
        """
        base_cost = self._total_cost_from_controls(scenario, x0, controls)
        controls_vec = controls.reshape(-1)
        grad = np.zeros_like(controls_vec)
        
        for idx in range(controls_vec.size):
            perturbed = controls_vec.copy()
            perturbed[idx] += self.config.finite_diff_eps
            perturbed_controls = perturbed.reshape(controls.shape)
            perturbed_cost = self._total_cost_from_controls(scenario, x0, perturbed_controls)
            grad[idx] = (perturbed_cost - base_cost) / self.config.finite_diff_eps
        
        return grad

    def _finite_difference_jacobian(
        self, scenario: Any, x0: np.ndarray, controls: np.ndarray, base_constraints: np.ndarray
    ) -> np.ndarray:
        """
        使用有限差分法计算约束函数关于控制输入的雅可比矩阵
        
        J_ij = ∂g_i/∂u_j ≈ (g_i(u + ε*e_j) - g_i(u)) / ε
        
        Args:
            scenario: 场景对象
            x0: 初始状态
            controls: 当前控制输入
            base_constraints: 当前约束值 g(u)
            
        Returns:
            雅可比矩阵 [n_constraints, nu*horizon]
        """
        controls_vec = controls.reshape(-1)
        jacobian = np.zeros((base_constraints.size, controls_vec.size), dtype=float)
        
        for idx in range(controls_vec.size):
            perturbed = controls_vec.copy()
            perturbed[idx] += self.config.finite_diff_eps
            perturbed_controls = perturbed.reshape(controls.shape)
            constraints = self._constraints_from_controls(scenario, x0, perturbed_controls)
            jacobian[:, idx] = (constraints - base_constraints) / self.config.finite_diff_eps
        
        return jacobian

    def _constraints_from_controls(
        self, scenario: Any, x0: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        """从控制输入计算约束值"""
        states = scenario.forward_simulation(x0, controls)
        return scenario.collect_constraints(states, controls)

    def _total_cost_from_controls(
        self, scenario: Any, x0: np.ndarray, controls: np.ndarray
    ) -> float:
        """从控制输入计算总成本"""
        states = scenario.forward_simulation(x0, controls)
        return scenario.total_cost(states, controls)

    def _kkt_residuals(
        self,
        grad: np.ndarray,
        jacobian: np.ndarray,
        constraints: np.ndarray,
        slack: np.ndarray,
        dual: np.ndarray,
        mu: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 KKT 残差
        
        KKT 条件：
        1. 对偶残差 (平稳性): r_dual = ∇f(u) - J(u)^T * λ
        2. 原始残差 (可行性): r_pri = g(u) + s
        3. 中心性残差 (互补松弛): r_cent = S*Λ*1 - μ*1
        
        Args:
            grad: 目标函数梯度 ∇f(u)
            jacobian: 约束雅可比矩阵 J(u)
            constraints: 约束值 g(u)
            slack: 松弛变量 s
            dual: 对偶变量 λ
            mu: 障碍参数
            
        Returns:
            (r_dual, r_pri, r_cent) 三类残差
        """
        # 对偶残差：平稳性条件
        r_dual = grad - jacobian.T @ dual
        
        # 原始残差：原始可行性条件（注意这里约束是 g(u) <= s，所以是 g(u) - s）
        r_pri = constraints - slack
        
        # 中心性残差：互补松弛条件的扰动形式
        r_cent = slack * dual - mu * np.ones_like(slack)
        
        return r_dual, r_pri, r_cent

    def _residual_norm(
        self, r_dual: np.ndarray, r_pri: np.ndarray, r_cent: np.ndarray
    ) -> float:
        """计算 KKT 残差的 L2 范数"""
        stacked = np.concatenate([r_dual, r_pri, r_cent], axis=0)
        return float(np.linalg.norm(stacked))

    def _solve_kkt_system(
        self,
        jacobian: np.ndarray,
        r_dual: np.ndarray,
        r_pri: np.ndarray,
        r_cent: np.ndarray,
        slack: np.ndarray,
        dual: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        求解 KKT 线性系统得到牛顿方向
        
        系统形式：
        [ H   -J^T   0  ] [Δu  ]   [-r_dual ]
        [ J    0    -I  ] [Δλ  ] = [-r_pri  ]
        [ 0    S    Λ  ] [Δs  ]   [-r_cent ]
        
        其中 H 使用阻尼单位阵近似：H = δ*I
        
        Args:
            jacobian: 约束雅可比矩阵
            r_dual: 对偶残差
            r_pri: 原始残差
            r_cent: 中心性残差
            slack: 当前松弛变量
            dual: 当前对偶变量
            
        Returns:
            (Δu, Δλ, Δs) 牛顿方向
        """
        n_vars = r_dual.size
        n_constraints = r_pri.size
        
        # 处理无约束情况
        if n_constraints == 0:
            hessian = np.eye(n_vars) * self.config.hessian_damping
            delta_u = -np.linalg.solve(hessian, r_dual)
            return delta_u, np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
        
        # 构造 Hessian 近似（阻尼单位阵）
        hessian = np.eye(n_vars) * self.config.hessian_damping
        
        # 构造 KKT 矩阵
        kkt_matrix = np.block(
            [
                [hessian, -jacobian.T, np.zeros((n_vars, n_constraints))],
                [jacobian, np.zeros((n_constraints, n_constraints)), -np.eye(n_constraints)],
                [np.zeros((n_constraints, n_vars)), np.diag(slack), np.diag(dual)],
            ]
        )
        
        # 构造右端项
        rhs = -np.concatenate([r_dual, r_pri, r_cent], axis=0)
        
        # 求解线性系统
        delta = np.linalg.solve(kkt_matrix, rhs)
        
        # 提取各变量的更新方向
        delta_u = delta[:n_vars]
        delta_dual = delta[n_vars : n_vars + n_constraints]
        delta_slack = delta[n_vars + n_constraints :]
        
        return delta_u, delta_dual, delta_slack

    def _line_search(
        self,
        slack: np.ndarray,
        dual: np.ndarray,
        delta_slack: np.ndarray,
        delta_dual: np.ndarray,
    ) -> float:
        """
        回溯线搜索，确保更新后的松弛变量和对偶变量保持正值
        
        寻找最大步长 α ∈ (0, 1] 使得：
            s + α*Δs > 0
            λ + α*Δλ > 0
        
        Args:
            slack: 当前松弛变量
            dual: 当前对偶变量
            delta_slack: 松弛变量的牛顿方向
            delta_dual: 对偶变量的牛顿方向
            
        Returns:
            步长 α
        """
        step = self.config.step_size
        
        for _ in range(20):
            if np.all(slack + step * delta_slack > 0.0) and np.all(
                dual + step * delta_dual > 0.0
            ):
                return step
            step *= self.config.step_decay
        
        return 0.0

    def _update_controls(
        self, controls: np.ndarray, delta_u: np.ndarray, step: float
    ) -> np.ndarray:
        """
        更新控制输入
        
        u_new = u + α * Δu
        
        Args:
            controls: 当前控制输入
            delta_u: 控制输入的牛顿方向
            step: 步长
            
        Returns:
            更新后的控制输入
        """
        updated = controls.reshape(-1) + step * delta_u
        return updated.reshape(controls.shape)
