"""
lq_approximation.py
====================
线性二次近似模块

功能：将非线性博弈（NonlinearGame）在当前操作点（Trajectory）处
      进行局部线性化/二次化，生成受约束的线性二次型博弈（ConstrainedLQGame）。

主要步骤：
  1. 动力学线性化  : f(x,u) ≈ A*x + B*u + c
  2. 约束线性化   : h(x,u) ≈ Hx*x + Hu*u + h
                    g(x,u) ≈ Gx*x + Gu*u + g
  3. 代价函数二次化: J_i ≈ 0.5*(Δx'Q_i Δx + Δu'R_i Δu + 2Δu'S_i Δx) + q_i'Δx + r_i'Δu
     其中二阶项还包含 Lagrangian 中动力学/约束对应的 Hessian 修正项。

使用数值微分（中心差分）计算 Jacobian 和 Hessian。
"""

import numpy as np

from .game_structs import ConstrainedLQGame, NonlinearGame, Trajectory


class LqApproximation:
    """
    线性二次近似类

    使用数值微分将非线性博弈局部线性化/二次化。
    对应 Julia 源码：src/lq_approximation.jl

    Parameters
    ----------
    eps_jac : float
        计算 Jacobian 的有限差分步长（中心差分），默认 1e-6。
    eps_hes : float
        计算 Hessian 的有限差分步长（中心差分），默认 1e-4。
    """

    def __init__(self, eps_jac: float = 1e-6, eps_hes: float = 1e-4):
        self.eps_jac = eps_jac
        self.eps_hes = eps_hes

    # ------------------------------------------------------------------
    # 私有数值微分方法
    # ------------------------------------------------------------------

    def _jacobian(self, f, z: np.ndarray) -> np.ndarray:
        """
        计算向量值函数 f: R^n -> R^m 的数值 Jacobian（中心差分）

        返回矩阵 J ∈ R^{m×n}，J[i,j] = ∂f_i/∂z_j
        """
        eps = self.eps_jac
        nz = len(z)
        fz = np.atleast_1d(f(z)).astype(float)
        mf = len(fz)
        J = np.zeros((mf, nz))
        for j in range(nz):
            zp = z.copy(); zp[j] += eps
            zm = z.copy(); zm[j] -= eps
            J[:, j] = (np.atleast_1d(f(zp)) - np.atleast_1d(f(zm))) / (2.0 * eps)
        return J

    def _hessian(self, f, z: np.ndarray) -> np.ndarray:
        """
        计算标量值函数 f: R^n -> R 的数值 Hessian（中心差分）

        返回对称矩阵 H ∈ R^{n×n}，H[i,j] = ∂²f/∂z_i∂z_j
        """
        eps = self.eps_hes
        nz = len(z)
        H = np.zeros((nz, nz))
        for i in range(nz):
            for j in range(i, nz):
                # 4点公式：(f(i+,j+) - f(i+,j-) - f(i-,j+) + f(i-,j-)) / (4ε²)
                zpp = z.copy(); zpp[i] += eps; zpp[j] += eps
                zpm = z.copy(); zpm[i] += eps; zpm[j] -= eps
                zmp = z.copy(); zmp[i] -= eps; zmp[j] += eps
                zmm = z.copy(); zmm[i] -= eps; zmm[j] -= eps
                hij = (f(zpp) - f(zpm) - f(zmp) + f(zmm)) / (4.0 * eps ** 2)
                H[i, j] = hij
                H[j, i] = hij
        return H

    def _gradient(self, f, z: np.ndarray) -> np.ndarray:
        """
        计算标量值函数 f: R^n -> R 的数值梯度（中心差分）

        返回向量 g ∈ R^n，g[i] = ∂f/∂z_i
        """
        eps = self.eps_jac
        nz = len(z)
        grad = np.zeros(nz)
        for i in range(nz):
            zp = z.copy(); zp[i] += eps
            zm = z.copy(); zm[i] -= eps
            grad[i] = (f(zp) - f(zm)) / (2.0 * eps)
        return grad

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def approximate(
        self,
        lq_approx: ConstrainedLQGame,
        g: NonlinearGame,
        current_op: Trajectory,
    ) -> None:
        """
        在当前操作点处对非线性博弈进行线性二次近似（原地修改 lq_approx）

        对应 Julia 源码中的 lq_approximation!(lq_approx, g, current_op)

        Parameters
        ----------
        lq_approx : ConstrainedLQGame
            将被就地更新的 LQ 近似对象（需已按正确尺寸预分配）。
        g : NonlinearGame
            原始非线性博弈。
        current_op : Trajectory
            当前操作点（包含状态轨迹和所有乘子）。
        """
        nx = g.nx
        nu = g.nu
        T = g.horizon
        l = g.equality_constraints_size
        ll = g.inequality_constraints_size

        # ================================================================
        # 阶段时刻 t = 0, 1, ..., T-1
        # ================================================================
        for t in range(T):
            xt = current_op.x[t]
            ut = current_op.u[t]
            # 拼接状态-控制向量 z = [x; u]
            z = np.concatenate([xt, ut])

            # ------------------------------------------------------------
            # 步骤 1：动力学线性化
            # f(x,u) ≈ A*x + B*u + c
            # ------------------------------------------------------------
            def f_z(z_, t_=t):
                return g.f_list[t_](z_[:nx], z_[nx: nx + nu])

            f_jac = self._jacobian(f_z, z)
            # A = ∂f/∂x
            lq_approx.A_list[t] = f_jac[:nx, :nx]
            # B = ∂f/∂u
            lq_approx.B_list[t] = f_jac[:nx, nx: nx + nu]
            # c = f(x_t,u_t) - x_{t+1}（动力学残差，用于修正线性化偏差）
            lq_approx.c_list[t] = (
                g.f_list[t](xt, ut) - current_op.x[t + 1]
            )

            # 预计算动力学 Hessian 列表（每个状态分量一个 Hessian）
            f_hessians = []
            for jj in range(nx):
                def f_jj_z(z_, jj_=jj, t_=t):
                    return float(g.f_list[t_](z_[:nx], z_[nx: nx + nu])[jj_])
                f_hessians.append(self._hessian(f_jj_z, z))

            for ii in range(g.n_players):
                # --------------------------------------------------------
                # 步骤 2：约束线性化
                # --------------------------------------------------------

                # 等式约束 h_i(x,u) ≈ Hx*x + Hu*u + h
                def h_func_z(z_, ii_=ii, t_=t):
                    return np.atleast_1d(
                        g.equality_constraints_list[t_][ii_](z_)
                    )

                h_jac = self._jacobian(h_func_z, z)
                lq_approx.Hx_list[t][ii] = h_jac[:l, :nx]
                lq_approx.Hu_list[t][ii] = h_jac[:l, nx: nx + nu]
                lq_approx.h_list[t][ii] = np.atleast_1d(
                    g.equality_constraints_list[t][ii](z)
                )

                # 不等式约束 g_i(x,u) ≈ Gx*x + Gu*u + g
                def g_func_z(z_, ii_=ii, t_=t):
                    return np.atleast_1d(
                        g.inequality_constraints_list[t_][ii_](z_)
                    )

                g_jac = self._jacobian(g_func_z, z)
                lq_approx.Gx_list[t][ii] = g_jac[:ll, :nx]
                lq_approx.Gu_list[t][ii] = g_jac[:ll, nx: nx + nu]
                lq_approx.g_list[t][ii] = np.atleast_1d(
                    g.inequality_constraints_list[t][ii](z)
                )

                # 等式约束 Hessian（每个约束分量）
                h_hessians = []
                for jj in range(l):
                    def h_jj_z(z_, ii_=ii, t_=t, jj_=jj):
                        return float(np.atleast_1d(
                            g.equality_constraints_list[t_][ii_](z_)
                        )[jj_])
                    h_hessians.append(self._hessian(h_jj_z, z))

                # 不等式约束 Hessian（每个约束分量）
                g_hessians = []
                for jj in range(ll):
                    def g_jj_z(z_, ii_=ii, t_=t, jj_=jj):
                        return float(np.atleast_1d(
                            g.inequality_constraints_list[t_][ii_](z_)
                        )[jj_])
                    g_hessians.append(self._hessian(g_jj_z, z))

                # --------------------------------------------------------
                # 步骤 3：代价函数二次化
                # 代价 Hessian 包含 Lagrangian 对应的修正项：
                # Q = ∂²J/∂x² + ∑_j λ_j ∂²f_j/∂x² - ∑_j μ_j ∂²h_j/∂x² - ∑_j γ_j ∂²g_j/∂x²
                # --------------------------------------------------------
                def cost_z(z_, ii_=ii, t_=t):
                    return float(g.costs_list[t_][ii_](z_[:nx], z_[nx: nx + nu]))

                cost_hess = self._hessian(cost_z, z)
                cost_grad = self._gradient(cost_z, z)

                # 提取当前时刻对应玩家的乘子
                lam_i = (
                    current_op.lam[t][ii * nx: (ii + 1) * nx]
                    if len(current_op.lam[t]) >= (ii + 1) * nx
                    else np.zeros(nx)
                )
                mu_i = (
                    current_op.mu[t][ii * l: (ii + 1) * l]
                    if current_op.mu and len(current_op.mu[t]) >= (ii + 1) * l
                    else np.zeros(l)
                )
                gamma_i = (
                    current_op.gamma[t][ii * ll: (ii + 1) * ll]
                    if current_op.gamma and len(current_op.gamma[t]) >= (ii + 1) * ll
                    else np.zeros(ll)
                )

                # Q_i = ∂²J_i/∂x² + Lagrangian 修正
                Q_ii = cost_hess[:nx, :nx].copy()
                for jj in range(nx):
                    Q_ii += f_hessians[jj][:nx, :nx] * lam_i[jj]
                for jj in range(l):
                    Q_ii -= h_hessians[jj][:nx, :nx] * mu_i[jj]
                for jj in range(ll):
                    Q_ii -= g_hessians[jj][:nx, :nx] * gamma_i[jj]
                lq_approx.Q_list[t][ii] = Q_ii

                # S_i = ∂²J_i/∂u∂x + Lagrangian 修正
                S_ii = cost_hess[nx: nx + nu, :nx].copy()
                for jj in range(nx):
                    S_ii += f_hessians[jj][nx: nx + nu, :nx] * lam_i[jj]
                for jj in range(l):
                    S_ii -= h_hessians[jj][nx: nx + nu, :nx] * mu_i[jj]
                for jj in range(ll):
                    S_ii -= g_hessians[jj][nx: nx + nu, :nx] * gamma_i[jj]
                lq_approx.S_list[t][ii] = S_ii

                # R_i = ∂²J_i/∂u² + Lagrangian 修正
                R_ii = cost_hess[nx: nx + nu, nx: nx + nu].copy()
                for jj in range(nx):
                    R_ii += f_hessians[jj][nx: nx + nu, nx: nx + nu] * lam_i[jj]
                for jj in range(l):
                    R_ii -= h_hessians[jj][nx: nx + nu, nx: nx + nu] * mu_i[jj]
                for jj in range(ll):
                    R_ii -= g_hessians[jj][nx: nx + nu, nx: nx + nu] * gamma_i[jj]
                lq_approx.R_list[t][ii] = R_ii

                # 代价函数线性项梯度 q_i = ∂J_i/∂x，r_i = ∂J_i/∂u
                lq_approx.q_list[t][ii] = cost_grad[:nx]
                lq_approx.r_list[t][ii] = cost_grad[nx: nx + nu]

        # ================================================================
        # 终端时刻 T（索引 T，即 Python 中 horizon）
        # ================================================================
        x_T = current_op.x[T]

        for ii in range(g.n_players):
            term_cost = g.terminal_costs_list[ii]
            cost_hess_T = self._hessian(term_cost, x_T)
            cost_grad_T = self._gradient(term_cost, x_T)

            # 终端等式约束 Jacobian
            def h_T_func(x_, ii_=ii):
                return np.atleast_1d(g.terminal_equality_constraints_list[ii_](x_))

            h_T_jac = self._jacobian(h_T_func, x_T)
            lq_approx.HxT[ii] = h_T_jac[:l, :nx]
            lq_approx.hxT[ii] = np.atleast_1d(
                g.terminal_equality_constraints_list[ii](x_T)
            )

            # 终端不等式约束 Jacobian
            def g_T_func(x_, ii_=ii):
                return np.atleast_1d(g.terminal_inequality_constraints_list[ii_](x_))

            g_T_jac = self._jacobian(g_T_func, x_T)
            lq_approx.GxT[ii] = g_T_jac[:ll, :nx]
            lq_approx.gxT[ii] = np.atleast_1d(
                g.terminal_inequality_constraints_list[ii](x_T)
            )

            # 终端等式约束 Hessian
            h_T_hessians = []
            for jj in range(l):
                def h_T_jj(x_, ii_=ii, jj_=jj):
                    return float(np.atleast_1d(
                        g.terminal_equality_constraints_list[ii_](x_)
                    )[jj_])
                h_T_hessians.append(self._hessian(h_T_jj, x_T))

            # 终端不等式约束 Hessian
            g_T_hessians = []
            for jj in range(ll):
                def g_T_jj(x_, ii_=ii, jj_=jj):
                    return float(np.atleast_1d(
                        g.terminal_inequality_constraints_list[ii_](x_)
                    )[jj_])
                g_T_hessians.append(self._hessian(g_T_jj, x_T))

            # 提取终端乘子
            mu_T = (
                current_op.mu[T][ii * l: (ii + 1) * l]
                if current_op.mu and len(current_op.mu) > T and len(current_op.mu[T]) >= (ii + 1) * l
                else np.zeros(l)
            )
            gamma_T = (
                current_op.gamma[T][ii * ll: (ii + 1) * ll]
                if current_op.gamma and len(current_op.gamma) > T and len(current_op.gamma[T]) >= (ii + 1) * ll
                else np.zeros(ll)
            )

            # 终端代价 Hessian（含 Lagrangian 修正）
            Q_T = cost_hess_T[:nx, :nx].copy()
            for jj in range(l):
                Q_T -= h_T_hessians[jj][:nx, :nx] * mu_T[jj]
            for jj in range(ll):
                Q_T -= g_T_hessians[jj][:nx, :nx] * gamma_T[jj]

            lq_approx.Q_list[T][ii] = Q_T
            lq_approx.q_list[T][ii] = cost_grad_T[:nx]
