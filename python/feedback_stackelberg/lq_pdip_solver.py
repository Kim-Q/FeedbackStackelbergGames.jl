"""
LQ 博弈的原始 - 对偶内点法 (PDIP) 求解器

基于 Julia 版本 pdip_fbst_lq_solver.jl 和 nw_pdip_fbst_lq_solver.jl 的 Python 实现
针对 LQ 博弈的特殊结构，使用解析推导向后递推求解策略矩阵

核心算法:
1. 对于两玩家 Stackelberg 博弈（玩家 1 是 leader，玩家 2 是 follower）
2. 在每个时刻 t，从 T 到 1 向后递推
3. 先求解 follower 的最优响应策略
4. 再求解 leader 的最优策略（考虑 follower 的响应）
5. 构造 KKT 矩阵并求解反馈增益矩阵
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from scipy.linalg import pinv


@dataclass
class LQGameParams:
    """LQ 博弈参数配置
    
    对应 Julia 中的 Constrained_LQGame struct
    """
    horizon: int  # 时间范围 T
    n_players: int = 2  # 玩家数量
    nx: int = 1  # 状态维度
    nu: int = 2  # 总控制维度
    players_u_index_list: Tuple[Tuple[int, ...], ...] = ((0,), (1,))  # 每个玩家的输入索引
    equality_constraints_size: int = 0  # 等式约束维度 l
    inequality_constraints_size: int = 0  # 不等式约束维度 ll
    
    # 系统动态参数 [A_list[t], B_list[t], c_list[t]] for t in 1:T
    A_list: List[np.ndarray] = field(default_factory=list)
    B_list: List[np.ndarray] = field(default_factory=list)
    c_list: List[np.ndarray] = field(default_factory=list)
    
    # 代价函数参数 [Q_list[t][player], S_list[t][player], R_list[t][player]] 
    # for t in 1:T+1 (Q), t in 1:T (S, R)
    Q_list: List[List[np.ndarray]] = field(default_factory=list)
    S_list: List[List[np.ndarray]] = field(default_factory=list)
    R_list: List[List[np.ndarray]] = field(default_factory=list)
    q_list: List[List[np.ndarray]] = field(default_factory=list)
    r_list: List[List[np.ndarray]] = field(default_factory=list)
    
    # 等式约束参数 Hx, Hu, h
    Hx_list: List[List[np.ndarray]] = field(default_factory=list)
    Hu_list: List[List[np.ndarray]] = field(default_factory=list)
    h_list: List[List[np.ndarray]] = field(default_factory=list)
    HxT: List[np.ndarray] = field(default_factory=list)
    hxT: List[np.ndarray] = field(default_factory=list)
    
    # 不等式约束参数 Gx, Gu, g
    Gx_list: List[List[np.ndarray]] = field(default_factory=list)
    Gu_list: List[List[np.ndarray]] = field(default_factory=list)
    g_list: List[List[np.ndarray]] = field(default_factory=list)
    GxT: List[np.ndarray] = field(default_factory=list)
    gxT: List[np.ndarray] = field(default_factory=list)
    
    # 初始状态
    x0: np.ndarray = field(default_factory=lambda: np.array([0.0]))


@dataclass
class Strategy:
    """策略数据结构
    
    对应 Julia 中的 strategy struct
    P[t]: 反馈增益矩阵 (nu x nx)
    α[t]: 偏移向量 (nu,)
    """
    P: List[np.ndarray]
    α: List[np.ndarray]


@dataclass
class Trajectory:
    """轨迹数据结构
    
    对应 Julia 中的 trajectory struct
    """
    x: List[np.ndarray]  # 状态轨迹
    u: List[np.ndarray]  # 控制轨迹
    λ: List[np.ndarray]  # 动力学拉格朗日乘子
    η: List[np.ndarray]  # 策略拉格朗日乘子
    ψ: List[np.ndarray] = field(default_factory=list)  # follower 策略乘子
    μ: List[np.ndarray] = field(default_factory=list)  # 等式约束乘子
    γ: List[np.ndarray] = field(default_factory=list)  # 不等式约束乘子
    s: List[np.ndarray] = field(default_factory=list)  # 松弛变量


class NWPDIPFBSTLQSolver:
    """
    Newton 原始 - 对偶内点法 Feedback Stackelberg LQ 求解器
    
    实现 Julia 版本 nw_pdip_fbst_lq_solver.jl 的 Python 版本
    使用解析推导的向后递推算法求解带约束的 LQ Stackelberg 博弈
    
    核心思想:
    1. 在每次外层迭代中，固定障碍参数 ρ
    2. 通过牛顿法求解 KKT 条件
    3. 利用 LQ 结构，向后递推计算反馈策略
    """
    
    def __init__(self, rho: float = 1.0, max_iter: int = 10, verbose: bool = True):
        """
        初始化求解器
        
        Args:
            rho: PDIP 惩罚参数 (ρ = 1/t)，设为 0.0 表示无不等式约束
            max_iter: 最大迭代次数
            verbose: 是否打印调试信息
        """
        self.rho = rho
        self.max_iter = max_iter
        self.verbose = verbose
    
    def solve(self, game: LQGameParams, current_op: Trajectory) -> Tuple[Trajectory, Strategy]:
        """
        求解 Feedback Stackelberg LQ 博弈
        
        Args:
            game: LQ 博弈参数
            current_op: 当前轨迹（包含初始的对偶变量估计）
            
        Returns:
            (trajectory, strategy): 最优轨迹和策略
        """
        # 提取问题维度
        l = game.equality_constraints_size
        ll = game.inequality_constraints_size
        nx, nu = game.nx, game.nu
        m = len(game.players_u_index_list[0])  # leader 的控制维度
        T = game.horizon
        num_player = game.n_players
        
        assert num_player == 2, "目前仅支持双玩家博弈"
        
        # 初始化策略
        strategies_P = [np.zeros((nu, nx)) for _ in range(T)]
        strategies_alpha = [np.zeros(nu) for _ in range(T)]
        
        # 初始化中间变量
        # 对偶变量
        gamma = [np.zeros(ll * num_player) for _ in range(T + 1)]
        mu = [np.zeros(l * num_player) for _ in range(T + 1)]
        lam = [np.zeros(nx * num_player) for _ in range(T)]
        eta = [np.zeros(nu) for _ in range(T - 1)]
        psi = [np.zeros(m) for _ in range(T)]
        
        # 检查是否存在非平凡的不等式约束
        have_intermediate_ineq = self._check_intermediate_ineq(game, ll)
        have_terminal_ineq = self._check_terminal_ineq(game, ll)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("NW-PDIP-FBST LQ 求解器启动")
            print(f"{'='*60}")
            print(f"Horizon T: {T}, 状态维度 nx: {nx}, 控制维度 nu: {nu}")
            print(f"Leader 控制维度 m: {m}, Follower 控制维度: {nu - m}")
            print(f"不等式约束维度 ll: {ll}, 等式约束维度 l: {l}")
            print(f"障碍参数 ρ: {self.rho}")
            print(f"存在中间不等式约束：{have_intermediate_ineq}")
            print(f"存在终端不等式约束：{have_terminal_ineq}")
            print(f"{'='*60}\n")
        
        # ========== 向后递推求解 ==========
        # 用于存储向后递推的增益矩阵
        K_next = np.zeros((nu, nx))
        k_next = np.zeros(nu)
        K_gamma_next = np.zeros((2 * ll, nx)) if ll > 0 else None
        k_gamma_next = np.zeros(2 * ll) if ll > 0 else None
        K_mu_next = np.zeros((2 * l, nx)) if l > 0 else None
        k_mu_next = np.zeros(2 * l) if l > 0 else None
        K_lambda_next = np.zeros((2 * nx, nx))
        k_lambda_next = np.zeros(2 * nx)
        
        # 存储 KKT 矩阵用于残差计算
        KKT_M_blocks = []
        KKT_N_blocks = []
        KKT_n_blocks = []
        
        for t in range(T, 0, -1):  # 从 T 到 1 向后递推
            if self.verbose:
                print(f"\n--- 时刻 t={t} ---")
            
            A = game.A_list[t - 1]
            B = game.B_list[t - 1]
            c = game.c_list[t - 1]
            
            # 提取各玩家的代价函数参数
            Q_next = np.zeros((nx * num_player, nx))
            R_t = np.zeros((nu, nu))
            q_next = np.zeros(nx * num_player)
            r_t = np.zeros(nu)
            
            for ii, udx in enumerate(game.players_u_index_list):
                idx_start = ii * nx
                idx_end = (ii + 1) * nx
                Q_next[idx_start:idx_end, :] = game.Q_list[t][ii]
                q_next[idx_start:idx_end] = game.q_list[t][ii]
                
                for j, u_idx in enumerate(udx):
                    R_t[u_idx, :] = game.R_list[t - 1][ii][u_idx, :]
                    r_t[u_idx] = game.r_list[t - 1][ii][u_idx]
            
            if t == T:
                # ===== 终端时刻 =====
                if self.verbose:
                    print("  处理终端时刻 t=T")
                
                # 先求解 follower (玩家 2)
                pi_hat_2, pi_check_2 = self._solve_follower_terminal(
                    game, t - 1, A, B, R_t, Q_next, r_t, q_next,
                    current_op.gamma[t] if ll > 0 else np.zeros(ll * num_player),
                    have_intermediate_ineq
                )
                
                # 再求解 leader (玩家 1)
                K, k = self._solve_leader_terminal(
                    game, t - 1, A, B, c, R_t, Q_next, r_t, q_next,
                    pi_hat_2, pi_check_2,
                    current_op.gamma[t] if ll > 0 else np.zeros(ll * num_player),
                    current_op.gamma[t + 1] if ll > 0 else np.zeros(ll * num_player),
                    have_intermediate_ineq, have_terminal_ineq
                )
                
                # 更新策略
                strategies_P[t - 1] = -K[:nu, :]
                strategies_alpha[t - 1] = -k[:nu].flatten()
                
                # 更新向后递推的增益
                K_next = -K[:nu, :]
                k_next = -k[:nu].flatten()
                
                if ll > 0:
                    K_gamma_next = -K[nu:nu + 2 * ll, :]
                    k_gamma_next = -k[nu:nu + 2 * ll].flatten()
                
                if l > 0:
                    K_mu_next = -K[nu + 2 * ll:nu + 2 * ll + 2 * l, :]
                    k_mu_next = -k[nu + 2 * ll:nu + 2 * ll + 2 * l].flatten()
                
                K_lambda_next = -K[nu + 2 * ll + 2 * l:nu + 2 * ll + 2 * l + 2 * nx, :]
                k_lambda_next = -k[nu + 2 * ll + 2 * l:nu + 2 * ll + 2 * l + 2 * nx].flatten()
                
            else:
                # ===== 中间时刻 t < T =====
                if self.verbose:
                    print(f"  处理中间时刻 t={t} < T")
                
                # 先求解 follower
                pi_hat_2, pi_check_2 = self._solve_follower_intermediate(
                    game, t - 1, A, B, R_t, Q_next, r_t, q_next,
                    K_next, k_next, K_gamma_next, k_gamma_next,
                    K_mu_next, k_mu_next, K_lambda_next, k_lambda_next,
                    current_op.gamma[t] if ll > 0 else np.zeros(ll * num_player),
                    current_op.gamma[t + 1] if ll > 0 else np.zeros(ll * num_player),
                    have_intermediate_ineq
                )
                
                # 再求解 leader
                K, k = self._solve_leader_intermediate(
                    game, t - 1, A, B, c, R_t, Q_next, r_t, q_next,
                    pi_hat_2, pi_check_2,
                    K_next, k_next, K_gamma_next, k_gamma_next,
                    K_mu_next, k_mu_next, K_lambda_next, k_lambda_next,
                    current_op.gamma[t] if ll > 0 else np.zeros(ll * num_player),
                    current_op.gamma[t + 1] if ll > 0 else np.zeros(ll * num_player),
                    have_intermediate_ineq
                )
                
                # 更新策略
                strategies_P[t - 1] = -K[:nu, :]
                strategies_alpha[t - 1] = -k[:nu].flatten()
                
                # 更新向后递推的增益
                K_next = -K[:nu, :]
                k_next = -k[:nu].flatten()
                
                if ll > 0:
                    K_gamma_next = -K[nu:nu + 2 * ll, :]
                    k_gamma_next = -k[nu:nu + 2 * ll].flatten()
                
                if l > 0:
                    K_mu_next = -K[nu + 2 * ll:nu + 2 * ll + 2 * l, :]
                    k_mu_next = -k[nu + 2 * ll + 2 * l:nu + 2 * l].flatten()
                
                K_lambda_next = -K[nu + 2 * ll + 2 * l:nu + 2 * ll + 2 * l + 2 * nx, :]
                k_lambda_next = -k[nu + 2 * ll + 2 * l:nu + 2 * ll + 2 * l + 2 * nx].flatten()
        
        # ========== 前向模拟 ==========
        if self.verbose:
            print(f"\n{'='*60}")
            print("前向模拟...")
        
        x_traj = [game.x0.copy()]
        u_traj = []
        
        x = game.x0.copy()
        for t in range(T):
            u = -strategies_P[t] @ x - strategies_alpha[t]
            u_traj.append(u.copy())
            x = game.A_list[t] @ x + game.B_list[t] @ u + game.c_list[t]
            x_traj.append(x.copy())
        
        # 构造轨迹对象
        trajectory = Trajectory(
            x=x_traj,
            u=u_traj,
            λ=lam,
            η=eta,
            ψ=psi,
            μ=mu,
            γ=gamma,
            s=current_op.s if hasattr(current_op, 's') else []
        )
        
        strategy = Strategy(P=strategies_P, α=strategies_alpha)
        
        if self.verbose:
            print(f"\n最终状态 x_T: {x_traj[-1]}")
            print(f"{'='*60}\n")
        
        return trajectory, strategy
    
    def _check_intermediate_ineq(self, game: LQGameParams, ll: int) -> bool:
        """检查是否存在非平凡的中间不等式约束"""
        if ll == 0:
            return False
        if len(game.Gx_list) == 0 or len(game.Gu_list) == 0 or len(game.g_list) == 0:
            return False
        
        for player in range(game.n_players):
            if np.linalg.norm(game.Gx_list[0][player]) > 1e-10:
                return True
            if np.linalg.norm(game.Gu_list[0][player]) > 1e-10:
                return True
            if np.linalg.norm(game.g_list[0][player]) > 1e-10:
                return True
        return False
    
    def _check_terminal_ineq(self, game: LQGameParams, ll: int) -> bool:
        """检查是否存在非平凡的终端不等式约束"""
        if ll == 0:
            return False
        if len(game.GxT) == 0 or len(game.gxT) == 0:
            return False
        
        for player in range(game.n_players):
            if np.linalg.norm(game.GxT[player]) > 1e-10:
                return True
            if np.linalg.norm(game.gxT[player]) > 1e-10:
                return True
        return False
    
    def _solve_follower_terminal(
        self, game: LQGameParams, t: int,
        A: np.ndarray, B: np.ndarray,
        R_t: np.ndarray, Q_next: np.ndarray,
        r_t: np.ndarray, q_next: np.ndarray,
        gamma_t: np.ndarray,
        have_ineq: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解终端时刻 follower 的最优响应
        
        返回 π̂² (依赖 x) 和 π̌² (依赖 u1)
        """
        nx, nu = game.nx, game.nu
        m = len(game.players_u_index_list[0])
        ll = game.inequality_constraints_size
        
        # 构造 follower 的 KKT 系统
        # 简化版本（无约束或约束不活跃时）
        M2 = np.block([
            [R_t[m:, m:], B[:, m:].T],
            [-B[:, m:], np.eye(nx)]
        ])
        
        N2 = np.block([
            [np.zeros((nu - m, nx)), np.zeros((nu - m, m))],
            [-A, -B[:, :m]]
        ])
        
        try:
            inv_M2_N2 = -pinv(M2) @ N2
            pi_hat_2 = inv_M2_N2[:nu - m, :nx]
            pi_check_2 = inv_M2_N2[:nu - m, nx:nx + m]
        except Exception as e:
            if self.verbose:
                print(f"  Warning: M2 奇异，使用正则化: {e}")
            M2_reg = M2 + 1e-6 * np.eye(M2.shape[0])
            inv_M2_N2 = -pinv(M2_reg) @ N2
            pi_hat_2 = inv_M2_N2[:nu - m, :nx]
            pi_check_2 = inv_M2_N2[:nu - m, nx:nx + m]
        
        return pi_hat_2, pi_check_2
    
    def _solve_leader_terminal(
        self, game: LQGameParams, t: int,
        A: np.ndarray, B: np.ndarray, c: np.ndarray,
        R_t: np.ndarray, Q_next: np.ndarray,
        r_t: np.ndarray, q_next: np.ndarray,
        pi_hat_2: np.ndarray, pi_check_2: np.ndarray,
        gamma_t: np.ndarray, gamma_T: np.ndarray,
        have_intermediate_ineq: bool,
        have_terminal_ineq: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解终端时刻 leader 的最优策略
        """
        nx, nu = game.nx, game.nu
        m = len(game.players_u_index_list[0])
        ll = game.inequality_constraints_size
        l = game.equality_constraints_size
        
        # 构造 leader 的 KKT 系统
        # 基础版本（无约束）
        Pi_2 = np.zeros((m, nu))
        Pi_2[:, m:] = pi_check_2
        
        Mt = np.block([
            [R_t, B.T, Pi_2.T],
            [-B, np.zeros((nx, nx)), np.eye(nx)],
            [np.zeros((nx, nu)), -np.eye(nx), Q_next]
        ])
        
        Nt = np.block([
            [np.zeros((nu, nx))],
            [-A],
            [np.zeros((nx, nx))]
        ])
        
        nt = np.block([
            r_t,
            -c,
            q_next
        ]).flatten()
        
        try:
            K = -pinv(Mt) @ Nt
            k = -pinv(Mt) @ nt
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Mt 奇异: {e}")
            Mt_reg = Mt + 1e-6 * np.eye(Mt.shape[0])
            K = -pinv(Mt_reg) @ Nt
            k = -pinv(Mt_reg) @ nt
        
        return K, k
    
    def _solve_follower_intermediate(
        self, game: LQGameParams, t: int,
        A: np.ndarray, B: np.ndarray,
        R_t: np.ndarray, Q_next: np.ndarray,
        r_t: np.ndarray, q_next: np.ndarray,
        K_next: np.ndarray, k_next: np.ndarray,
        K_gamma_next: Optional[np.ndarray], k_gamma_next: Optional[np.ndarray],
        K_mu_next: Optional[np.ndarray], k_mu_next: Optional[np.ndarray],
        K_lambda_next: np.ndarray, k_lambda_next: np.ndarray,
        gamma_t: np.ndarray, gamma_next: np.ndarray,
        have_ineq: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解中间时刻 follower 的最优响应
        """
        nx, nu = game.nx, game.nu
        m = len(game.players_u_index_list[0])
        ll = game.inequality_constraints_size
        
        # 构造 follower 的 KKT 系统
        # 考虑向后递推的价值函数
        Q_eff = Q_next[nx:, :] - A.T @ K_lambda_next[nx:, :]
        
        M2 = np.block([
            [R_t[m:, m:], B[:, m:].T],
            [-B[:, m:], np.eye(nx)]
        ])
        
        N2 = np.block([
            [np.zeros((nu - m, nx)), np.zeros((nu - m, m))],
            [-A, -B[:, :m]]
        ])
        
        try:
            inv_M2_N2 = -pinv(M2) @ N2
            pi_hat_2 = inv_M2_N2[:nu - m, :nx]
            pi_check_2 = inv_M2_N2[:nu - m, nx:nx + m]
        except Exception as e:
            if self.verbose:
                print(f"  Warning: M2 奇异: {e}")
            M2_reg = M2 + 1e-6 * np.eye(M2.shape[0])
            inv_M2_N2 = -pinv(M2_reg) @ N2
            pi_hat_2 = inv_M2_N2[:nu - m, :nx]
            pi_check_2 = inv_M2_N2[:nu - m, nx:nx + m]
        
        return pi_hat_2, pi_check_2
    
    def _solve_leader_intermediate(
        self, game: LQGameParams, t: int,
        A: np.ndarray, B: np.ndarray, c: np.ndarray,
        R_t: np.ndarray, Q_next: np.ndarray,
        r_t: np.ndarray, q_next: np.ndarray,
        pi_hat_2: np.ndarray, pi_check_2: np.ndarray,
        K_next: np.ndarray, k_next: np.ndarray,
        K_gamma_next: Optional[np.ndarray], k_gamma_next: Optional[np.ndarray],
        K_mu_next: Optional[np.ndarray], k_mu_next: Optional[np.ndarray],
        K_lambda_next: np.ndarray, k_lambda_next: np.ndarray,
        gamma_t: np.ndarray, gamma_next: np.ndarray,
        have_ineq: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解中间时刻 leader 的最优策略
        """
        nx, nu = game.nx, game.nu
        m = len(game.players_u_index_list[0])
        ll = game.inequality_constraints_size
        
        # 构造 leader 的 KKT 系统
        Pi_2 = np.zeros((m, nu))
        Pi_2[:, m:] = pi_check_2
        
        Pi_next = np.zeros((nu, nx * 2))
        Pi_next[:m, :nx] = pi_hat_2
        Pi_next[m:, nx:] = K_next
        
        Q_eff = Q_next - K_lambda_next.T @ A
        
        Mt = np.block([
            [R_t, B.T, Pi_2.T],
            [-B, np.zeros((nx, nx)), np.eye(nx)],
            [np.zeros((nx, nu)), -np.eye(nx * 2), Q_eff]
        ])
        
        Nt = np.block([
            [np.zeros((nu, nx))],
            [-A],
            [np.zeros((nx * 2, nx))]
        ])
        
        nt = np.block([
            r_t,
            -c,
            q_next
        ]).flatten()
        
        try:
            K = -pinv(Mt) @ Nt
            k = -pinv(Mt) @ nt
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Mt 奇异: {e}")
            Mt_reg = Mt + 1e-6 * np.eye(Mt.shape[0])
            K = -pinv(Mt_reg) @ Nt
            k = -pinv(Mt_reg) @ nt
        
        return K, k


def create_lq_game_from_scenario(scenario, params) -> LQGameParams:
    """
    从 scenario 对象创建 LQGameParams
    
    用于将 LQRScenario 转换为 LQGameParams 格式
    
    注意：Julia 版本中 Q_list[t] 对应时刻 t 的代价，其中 t=1..T+1
    Python 中使用 0-based indexing，所以 Q_list[t] 对应时刻 t (0-based)
    """
    nx = scenario.nx
    nu = scenario.nu
    m = scenario._nu_leader
    T = params.horizon
    
    # 构造 LQ 参数
    game = LQGameParams(
        horizon=T,
        n_players=2,
        nx=nx,
        nu=nu,
        players_u_index_list=(tuple(range(m)), tuple(range(m, nu))),
        equality_constraints_size=0,
        inequality_constraints_size=0,
        x0=params.x0.copy()
    )
    
    # 填充动态参数 (t=0..T-1 对应 Julia 的 t=1..T)
    game.A_list = [params.A.copy() for _ in range(T)]
    game.B_list = [np.hstack([params.B1, params.B2]) for _ in range(T)]
    game.c_list = [np.zeros(nx) for _ in range(T)]
    
    # 填充代价函数参数
    # Q_list 有 T+1 个元素（包括终端），对应 Julia 的 t=1..T+1
    # 在 Python 中索引为 0..T，其中索引 T 是终端代价
    game.Q_list = []
    game.R_list = []
    game.S_list = []
    game.q_list = []
    game.r_list = []
    
    for t in range(T):
        # 阶段代价 (t=0..T-1)
        Q_t = [params.Q1.copy(), params.Q2.copy()]
        # R 矩阵需要正确构造
        # 玩家 1 (leader): J1 = 0.5 * [x'Q1*x + u1'R11*u1 + u2'Theta1*u2 + u2'R12*u2]
        # 玩家 2 (follower): J2 = 0.5 * [x'Q2*x + u2'R22*u2 + u1'Theta2*u1 + u1'R21*u1]
        R1 = np.block([
            [params.R11, np.zeros((m, nu - m))],
            [np.zeros((nu - m, m)), params.R12 + params.Theta1]
        ])
        R2 = np.block([
            [params.Theta2 + params.R21, np.zeros((m, nu - m))],
            [np.zeros((nu - m, m)), params.R22]
        ])
        R_t = [R1, R2]
        
        S_t = [np.zeros((nu, nx)), np.zeros((nu, nx))]
        q_t = [np.zeros(nx), np.zeros(nx)]
        r_t = [np.zeros(nu), np.zeros(nu)]
        
        game.Q_list.append(Q_t)
        game.R_list.append(R_t)
        game.S_list.append(S_t)
        game.q_list.append(q_t)
        game.r_list.append(r_t)
    
    # 终端代价 (索引 T 对应 Julia 的 T+1)
    game.Q_list.append([params.Q_terminal_leader.copy(), params.Q_terminal_follower.copy()])
    game.q_list.append([np.zeros(nx), np.zeros(nx)])
    
    return game
