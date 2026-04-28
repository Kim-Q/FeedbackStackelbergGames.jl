from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from feedback_stackelberg.config import PDIPConfig


@dataclass
class PDIPResult:
    states: np.ndarray
    controls: np.ndarray
    slack: np.ndarray
    dual: np.ndarray
    loss_history: List[List[float]]
    residual_history: List[List[float]]


class PDIPSolver:
    def __init__(self, config: PDIPConfig | None = None):
        self.config = config or PDIPConfig()

    def solve(self, scenario: Any, initial_state: np.ndarray | None = None) -> PDIPResult:
        x0 = initial_state.copy() if initial_state is not None else scenario.initial_state()
        controls = np.zeros((scenario.params.horizon, scenario.nu), dtype=float)
        constraint_dim = scenario.collect_constraints(
            scenario.forward_simulation(x0, controls), controls
        ).shape[0]
        slack = np.ones(constraint_dim, dtype=float) * self.config.initial_slack
        dual = np.ones(constraint_dim, dtype=float) * self.config.initial_dual
        loss_history: List[List[float]] = []
        residual_history: List[List[float]] = []
        mu = self.config.barrier_mu

        for outer_iter in range(self.config.outer_iter):
            inner_loss: List[float] = []
            inner_residual: List[float] = []
            for iteration in range(self.config.max_iter):
                # 步骤1：前向模拟得到状态轨迹
                states = scenario.forward_simulation(x0, controls)
                # 步骤2：计算约束与目标函数
                constraints = scenario.collect_constraints(states, controls)
                total_cost = scenario.total_cost(states, controls)
                # 步骤3：构造KKT残差（原始、对偶与互补残差）
                grad = self._finite_difference_gradient(scenario, x0, controls)
                jacobian = self._finite_difference_jacobian(scenario, x0, controls, constraints)
                r_dual, r_pri, r_cent = self._kkt_residuals(grad, jacobian, constraints, slack, dual, mu)
                residual_norm = self._residual_norm(r_dual, r_pri, r_cent)
                inner_loss.append(total_cost)
                inner_residual.append(residual_norm)
                if residual_norm < self.config.residual_tol:
                    break
                # 步骤4：求解牛顿方向
                delta_u, delta_dual, delta_slack = self._solve_kkt_system(
                    jacobian, r_dual, r_pri, r_cent, slack, dual
                )
                # 步骤5：线搜索并保证正性约束
                step = self._line_search(slack, dual, delta_slack, delta_dual)
                # 步骤6：更新原始变量与对偶变量
                controls = self._update_controls(controls, delta_u, step)
                slack = slack + step * delta_slack
                dual = dual + step * delta_dual
            loss_history.append(inner_loss)
            residual_history.append(inner_residual)
            mu *= self.config.barrier_decay

        states = scenario.forward_simulation(x0, controls)
        return PDIPResult(
            states=states,
            controls=controls,
            slack=slack,
            dual=dual,
            loss_history=loss_history,
            residual_history=residual_history,
        )

    def _finite_difference_gradient(self, scenario: Any, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
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
        controls_vec = controls.reshape(-1)
        jacobian = np.zeros((base_constraints.size, controls_vec.size), dtype=float)
        for idx in range(controls_vec.size):
            perturbed = controls_vec.copy()
            perturbed[idx] += self.config.finite_diff_eps
            perturbed_controls = perturbed.reshape(controls.shape)
            constraints = self._constraints_from_controls(scenario, x0, perturbed_controls)
            jacobian[:, idx] = (constraints - base_constraints) / self.config.finite_diff_eps
        return jacobian

    def _constraints_from_controls(self, scenario: Any, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = scenario.forward_simulation(x0, controls)
        return scenario.collect_constraints(states, controls)

    def _total_cost_from_controls(self, scenario: Any, x0: np.ndarray, controls: np.ndarray) -> float:
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
        r_dual = grad - jacobian.T @ dual
        r_pri = constraints - slack
        r_cent = slack * dual - mu * np.ones_like(slack)
        return r_dual, r_pri, r_cent

    def _residual_norm(self, r_dual: np.ndarray, r_pri: np.ndarray, r_cent: np.ndarray) -> float:
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
        n_vars = r_dual.size
        n_constraints = r_pri.size
        if n_constraints == 0:
            hessian = np.eye(n_vars) * self.config.hessian_damping
            delta_u = -np.linalg.solve(hessian, r_dual)
            return delta_u, np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
        hessian = np.eye(n_vars) * self.config.hessian_damping
        kkt_matrix = np.block(
            [
                [hessian, -jacobian.T, np.zeros((n_vars, n_constraints))],
                [jacobian, np.zeros((n_constraints, n_constraints)), -np.eye(n_constraints)],
                [np.zeros((n_constraints, n_vars)), np.diag(slack), np.diag(dual)],
            ]
        )
        rhs = -np.concatenate([r_dual, r_pri, r_cent], axis=0)
        delta = np.linalg.solve(kkt_matrix, rhs)
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
        step = self.config.step_size
        for _ in range(20):
            if np.all(slack + step * delta_slack > 0.0) and np.all(dual + step * delta_dual > 0.0):
                return step
            step *= self.config.step_decay
        return 0.0

    def _update_controls(self, controls: np.ndarray, delta_u: np.ndarray, step: float) -> np.ndarray:
        updated = controls.reshape(-1) + step * delta_u
        return updated.reshape(controls.shape)
