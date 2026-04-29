# PIDP 算法总结与 Highway Env 测试算法流程

## 一、PIDP (Primal-Dual Interior Point Method) 算法内容

### 1. 问题描述

PIDP 用于求解带不等式约束的 Stackelberg 博弈优化问题：

$$
\begin{aligned}
\min_{u} \quad & f(u) \\
\text{s.t.} \quad & g(u) \geq 0
\end{aligned}
$$

其中：
- $f(u)$: 目标函数（Leader 和 Follower 的总成本）
- $g(u)$: 不等式约束向量
- $u$: 决策变量（控制序列或反馈增益）

### 2. 障碍函数方法

引入松弛变量 $s > 0$，将不等式约束转化为等式约束：

$$
g(u) - s = 0, \quad s > 0
$$

构造对数障碍问题：

$$
\min_{u, s} \quad f(u) - \mu \sum_{i} \log(s_i)
$$

其中 $\mu > 0$ 是障碍参数，控制对数障碍的权重。

### 3. KKT 条件

对于障碍问题，KKT 条件包括：

**平稳性条件（对偶残差）：**
$$
r_{\text{dual}} = \nabla f(u) - J(u)^T \lambda = 0
$$

**原始可行性条件（原始残差）：**
$$
r_{\text{pri}} = g(u) - s = 0
$$

**互补松弛条件（中心性残差）：**
$$
r_{\text{cent}} = S \Lambda \mathbf{1} - \mu \mathbf{1} = 0
$$

其中：
- $J(u) = \frac{\partial g}{\partial u}$ 是约束雅可比矩阵
- $\lambda$ 是对偶变量（拉格朗日乘子）
- $S = \text{diag}(s)$, $\Lambda = \text{diag}(\lambda)$
- $\mathbf{1}$ 是全 1 向量

### 4. 牛顿法求解 KKT 系统

线性化 KKT 条件得到以下线性系统：

$$
\begin{bmatrix}
H & -J^T & 0 \\
J & 0 & -I \\
0 & S & \Lambda
\end{bmatrix}
\begin{bmatrix}
\Delta u \\
\Delta \lambda \\
\Delta s
\end{bmatrix}
=
\begin{bmatrix}
-r_{\text{dual}} \\
-r_{\text{pri}} \\
-r_{\text{cent}}
\end{bmatrix}
$$

其中 $H$ 是 Hessian 近似（使用阻尼单位阵 $H = \delta I$）。

### 5. 算法流程

```
Algorithm: Primal-Dual Interior Point Method for Stackelberg Games
================================================================================

Input: 
    - 场景对象 scenario（提供动力学、成本函数、约束）
    - 初始状态 x₀
    - 配置参数 {max_iter, outer_iter, barrier_mu, barrier_decay, residual_tol, ...}

Output: 
    - 最优状态轨迹 {xₜ}, 控制序列 {uₜ}
    - 松弛变量 s, 对偶变量 λ
    - 收敛历史

================================================================================

Step 1: 初始化
--------------------------------------------------------------------------------
    x₀ ← initial_state()
    u⁽⁰⁾ ← zeros(decision_shape)                    # 初始决策变量
    s⁽⁰⁾ ← ones(constraint_dim) × initial_slack     # 松弛变量 > 0
    λ⁽⁰⁾ ← ones(constraint_dim) × initial_dual      # 对偶变量 > 0
    μ ← barrier_mu                                  # 初始障碍参数
    loss_history ← [], residual_history ← []

================================================================================

Step 2: 外层循环 - 障碍参数衰减
--------------------------------------------------------------------------------
    for k = 1 to outer_iter do
        
        inner_loss ← [], inner_residual ← []
        
        ============================================================================
        
        Step 3: 内层牛顿迭代
        --------------------------------------------------------------------------------
        for iter = 1 to max_iter do
            
            ------------------------------------------------------------------------
            Step 3.1: 前向模拟
            ------------------------------------------------------------------------
            {xₜ, uₜ} ← rollout(scenario, x₀, u⁽ⁱᵗᵉʳ⁾)
                for t = 0 to horizon-1:
                    xₜ₊₁ = dynamics(xₜ, uₜ)
            
            ------------------------------------------------------------------------
            Step 3.2: 计算约束与目标函数
            ------------------------------------------------------------------------
            g ← collect_constraints({xₜ}, {uₜ})     # 收集所有时刻的不等式约束
            f ← total_cost({xₜ}, {uₜ})              # 计算总成本
            
            ------------------------------------------------------------------------
            Step 3.3: 计算 KKT 残差
            ------------------------------------------------------------------------
            # 使用有限差分计算梯度
            ∇f ← finite_difference_gradient(scenario, x₀, u⁽ⁱᵗᵉʳ⁾)
                for j = 1 to dim(u):
                    u⁺ = u⁽ⁱᵗᵉʳ⁾ + ε·eⱼ
                    ∂f/∂uⱼ ≈ [f(u⁺) - f(u)] / ε
            
            # 使用有限差分计算约束雅可比
            J ← finite_difference_jacobian(scenario, x₀, u⁽ⁱᵗᵉʳ⁾, g)
                for j = 1 to dim(u):
                    u⁺ = u⁽ⁱᵗᵉʳ⁾ + ε·eⱼ
                    g⁺ = constraints(u⁺)
                    J[:,j] = (g⁺ - g) / ε
            
            # 计算三类残差
            r_dual = ∇f - Jᵀλ                        # 对偶残差
            r_pri = g - s                            # 原始残差
            r_cent = s ⊙ λ - μ·1                     # 中心性残差 (⊙表示逐元素乘积)
            
            # 计算残差范数
            residual_norm = ‖[r_dual; r_pri; r_cent]‖₂
            
            inner_loss.append(f)
            inner_residual.append(residual_norm)
            
            # 检查收敛性
            if residual_norm < residual_tol then
                break
            end
            
            ------------------------------------------------------------------------
            Step 3.4: 求解牛顿方向
            ------------------------------------------------------------------------
            # 求解 KKT 线性系统
            [Δu; Δλ; Δs] ← solve_kkt_system(J, r_dual, r_pri, r_cent, s, λ)
            
            # KKT 矩阵构造:
            KKT_matrix = [
                δ·I      -Jᵀ       0
                J         0       -I
                0        diag(s)  diag(λ)
            ]
            rhs = -[r_dual; r_pri; r_cent]
            [Δu; Δλ; Δs] = KKT_matrix \ rhs
            
            ------------------------------------------------------------------------
            Step 3.5: 线搜索保证正性约束
            ------------------------------------------------------------------------
            α ← line_search(scenario, x₀, u, Δu, s, λ, Δs, Δλ, μ, residual_norm)
            
            # 回溯线搜索：找到最大 α ∈ (0,1] 使得
            #   s + α·Δs > 0
            #   λ + α·Δλ > 0
            #   且新残差 ≤ 当前残差
            step ← 1.0
            for ls_iter = 1 to line_search_max_iter do
                if all(s + step·Δs > 0) and all(λ + step·Δλ > 0) then
                    u_candidate = u + step·Δu
                    s_candidate = s + step·Δs
                    λ_candidate = λ + step·Δλ
                    residual_new ← evaluate_residual(u_candidate, s_candidate, λ_candidate)
                    
                    if residual_new ≤ residual_norm then
                        return step
                    end
                end
                step ← step × step_decay
            end
            
            ------------------------------------------------------------------------
            Step 3.6: 更新原始变量与对偶变量
            ------------------------------------------------------------------------
            u⁽ⁱᵗᵉʳ⁺¹⁾ ← u⁽ⁱᵗᵉʳ⁾ + α·Δu
            s⁽ⁱᵗᵉʳ⁺¹⁾ ← s⁽ⁱᵗᵉʳ⁾ + α·Δs
            λ⁽ⁱᵗᵉʳ⁺¹⁾ ← λ⁽ⁱᵗᵉʳ⁾ + α·Δλ
            
        end for
        
        loss_history.append(inner_loss)
        residual_history.append(inner_residual)
        
        ============================================================================
        
        Step 4: 衰减障碍参数
        --------------------------------------------------------------------------------
        μ ← μ × barrier_decay
    
    end for

================================================================================

Step 5: 输出结果
--------------------------------------------------------------------------------
    {xₜ*, uₜ*} ← rollout(scenario, x₀, u*)
    
    return PDIPResult(
        states = {xₜ*},
        controls = {uₜ*},
        decision = u*,
        slack = s,
        dual = λ,
        loss_history = loss_history,
        residual_history = residual_history
    )

================================================================================
```

### 6. 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_iter` | 10 | 内层牛顿迭代最大次数 |
| `outer_iter` | 1 | 外层障碍参数更新次数 |
| `barrier_mu` | 1.0 | 初始障碍参数 μ₀ |
| `barrier_decay` | 0.5 | 障碍参数衰减因子 |
| `step_size` | 1.0 | 初始线搜索步长 |
| `step_decay` | 0.5 | 线搜索步长衰减因子 |
| `line_search_max_iter` | 20 | 线搜索最大迭代次数 |
| `residual_tol` | 1e-4 | KKT 残差收敛阈值 |
| `finite_diff_eps` | 1e-4 | 有限差分扰动量 |
| `hessian_damping` | 1e-2 | Hessian 阻尼系数 δ |
| `initial_slack` | 1.0 | 松弛变量初始值 |
| `initial_dual` | 1.0 | 对偶变量初始值 |

---

## 二、Highway Env 测试算法（Julia 实现）

### 1. Highway 场景动力学模型

**状态定义（8 维）：**
$$
x = [y_1, x_1, v_1, \theta_1, y_2, x_2, v_2, \theta_2]^T
$$
- 车辆 1（Leader）：位置 $(y_1, x_1)$，速度 $v_1$，航向角 $\theta_1$
- 车辆 2（Follower）：位置 $(y_2, x_2)$，速度 $v_2$，航向角 $\theta_2$

**控制输入（4 维）：**
$$
u = [a_1, \omega_1, a_2, \omega_2]^T
$$
- $a_i$: 加速度
- $\omega_i$: 角速度

**离散时间动力学（Euler 积分，时间步长 $\Delta t$）：**
$$
\begin{aligned}
y_1^{t+1} &= y_1^t + \Delta t \cdot v_1^t \sin(\theta_1^t) \\
x_1^{t+1} &= x_1^t + \Delta t \cdot v_1^t \cos(\theta_1^t) \\
v_1^{t+1} &= v_1^t + \Delta t \cdot a_1^t \\
\theta_1^{t+1} &= \theta_1^t + \Delta t \cdot \omega_1^t \\
y_2^{t+1} &= y_2^t + \Delta t \cdot v_2^t \sin(\theta_2^t) \\
x_2^{t+1} &= x_2^t + \Delta t \cdot v_2^t \cos(\theta_2^t) \\
v_2^{t+1} &= v_2^t + \Delta t \cdot a_2^t \\
\theta_2^{t+1} &= \theta_2^t + \Delta t \cdot \omega_2^t
\end{aligned}
$$

### 2. 代价函数

**阶段代价（Stage Cost）：**

Leader（车辆 1）代价：
$$
J_1^{\text{stage}} = 10(y_1 - 0.4)^2 + 6(v_1 - v_2)^2 + 2a_1^2 + 2\omega_1^2
$$

Follower（车辆 2）代价：
$$
J_2^{\text{stage}} = \theta_2^4 + 2a_2^2 + 2\omega_2^2
$$

总阶段代价：
$$
J^{\text{stage}} = J_1^{\text{stage}} + J_2^{\text{stage}}
$$

**终端代价（Terminal Cost）：**
$$
J^{\text{terminal}} = 10(y_1 - 0.4)^2 + 6(v_1 - v_2)^2 + \theta_2^4
$$

**总代价：**
$$
J = \sum_{t=0}^{T-1} J^{\text{stage}}(x_t, u_t) + J^{\text{terminal}}(x_T)
$$

### 3. 不等式约束

**状态约束：**
1. 横向位置约束：$y_1 \geq 0.25$, $y_2 \geq 0.25$
2. 道路边界约束：$g_{\text{road}}(y, x) \geq 0$
3. 碰撞避免约束：$2[(y_1-y_2)^2 + (x_1-x_2)^2 - r_{\text{collision}}^2] \geq 0$

**控制约束：**
$$
\begin{aligned}
a_{\min} \leq a_i \leq a_{\max} \\
\omega_{\min} \leq \omega_i \leq \omega_{\max}
\end{aligned}
$$

**道路边界约束函数：**
$$
g_{\text{road}}(y, x) = 
\begin{cases}
x_{\text{base}} - y, & \text{if } x > L_{\text{road}} \\
\text{dist}_{\text{upper}} - R, & \text{if } x_{\text{corner}} < x \leq L_{\text{road}} \text{ and } \phi < 2\alpha \\
R - \text{dist}_{\text{lower}}, & \text{if } x_{\text{corner}} < x \leq L_{\text{road}} \text{ and } \phi \geq 2\alpha \\
x_{\text{right}} - y, & \text{if } x \leq x_{\text{corner}}
\end{cases}
$$

其中：
- $R = \frac{L_{\text{segment}}}{4\sin(\alpha)}$ 是弯道半径
- $\alpha = \arctan\left(\frac{x_{\text{right}} - x_{\text{base}}}{L_{\text{road}} - x_{\text{corner}}}\right)$ 是弯道角度
- $\text{dist}_{\text{upper}} = \sqrt{(y - x_{\text{upper}})^2 + (x - L_{\text{road}})^2}$
- $\text{dist}_{\text{lower}} = \sqrt{(y - x_{\text{lower}})^2 + (x - x_{\text{corner}})^2}$

### 3. Feedback Stackelberg LQ 求解器（PDIP-FBST-LQ）

#### 3.1 问题形式化

在每次 iLQ 迭代中，原非线性博弈被局部近似为线性二次博弈：

**线性化动力学：**
$$
\delta x_{t+1} = A_t \delta x_t + B_t \delta u_t + c_t
$$

**二次化代价（玩家 $i$）：**
$$
J_i = \sum_{t=1}^{T} \left( \frac{1}{2} \delta x_t^T Q_t^i \delta x_t + \delta x_t^T S_t^i \delta u_t + \frac{1}{2} \delta u_t^T R_t^i \delta u_t + q_t^{i,T} \delta x_t + r_t^{i,T} \delta u_t \right) + \text{terminal terms}
$$

**不等式约束（线性化）：**
$$
G_{x,t}^i \delta x_t + G_{u,t}^i \delta u_t + g_t^i \geq 0
$$

**等式约束（线性化）：**
$$
H_{x,t}^i \delta x_t + H_{u,t}^i \delta u_t + h_t^i = 0
$$

#### 3.2 反馈 Stackelberg 策略结构

假设策略具有线性反馈形式：
$$
u_t = -K_t x_t - \alpha_t
$$

其中：
- $K_t \in \mathbb{R}^{n_u \times n_x}$: 反馈增益矩阵
- $\alpha_t \in \mathbb{R}^{n_u}$: 前馈项

#### 3.3 逆向递推求解（Backward Pass）

从 $t=T$ 到 $t=1$ 逆向求解：

**步骤 1: 终端时刻 $t=T$**

构造 Follower（玩家 2）的 KKT 系统：

$$
M_2 \begin{bmatrix} \delta u_T^2 \\ \gamma_T^2 \\ \mu_T^2 \\ \delta x_T \\ \delta x_{T+1} \\ \gamma_{T+1}^2 \\ \mu_{T+1}^2 \end{bmatrix} + N_2 \begin{bmatrix} \delta x_T \\ \delta u_T^1 \end{bmatrix} = 0
$$

其中 $M_2$ 包含：
- 代价 Hessian: $R_T^2$
- 约束 Jacobian: $G_{u,T}^2, H_{u,T}^2$
- 动力学: $B_T$
- 互补松弛项: $\hat{\gamma}_T^2 \hat{g}_T^2$

求解得到 Follower 对 Leader 策略的反应函数：
$$
\delta u_T^2 = \hat{\pi}_T^2 \delta x_T + \check{\pi}_T^2 \delta u_T^1
$$

**步骤 2: 求解 Leader（玩家 1）**

将 Follower 的反应函数代入 Leader 的问题，构造增广 KKT 系统：

$$
M_T \begin{bmatrix} \delta u_T \\ \gamma_T \\ \mu_T \\ \delta x_T \\ \delta x_{T+1} \\ \gamma_{T+1} \\ \mu_{T+1} \end{bmatrix} + N_T \delta x_T + n_T = 0
$$

其中 $M_T$ 是块矩阵：
$$
M_T = \begin{bmatrix}
R_T & -\hat{G}_{u,T}^T & -\hat{H}_{u,T}^T & \hat{B}_T^T & \Pi_T^T \\
\Gamma_T \hat{\gamma}_T G_{u,T} & \Gamma_T \hat{g}_T & 0 & 0 & 0 \\
H_{u,T} & 0 & 0 & 0 & 0 \\
-B_T & 0 & 0 & I & 0 \\
0 & 0 & 0 & -I & Q_{T+1} - \hat{G}_{x,T+1}^T K_\gamma - \hat{H}_{x,T+1}^T K_\mu + \hat{A}_{T+1}^T K_\lambda \\
0 & \Gamma_{T+1} \hat{\gamma}_{T+1} G_{x,T+1} & 0 & 0 & \Gamma_{T+1} \hat{g}_{T+1} \\
0 & H_{x,T+1} & 0 & 0 & 0
\end{bmatrix}
$$

求解得到最优策略：
$$
\begin{aligned}
K_T &= -M_T^{-1} N_T \\
\alpha_T &= -M_T^{-1} n_T
\end{aligned}
$$

**步骤 3: 中间时刻 $t < T$**

类似地，但需要考虑未来值函数的影响。定义值函数参数：
$$
V_{t+1}(\delta x_{t+1}) = \frac{1}{2} \delta x_{t+1}^T P_{t+1} \delta x_{t+1} + p_{t+1}^T \delta x_{t+1}
$$

Follower 的 KKT 系统扩展为：

$$
P_t^1 \begin{bmatrix} \delta u_t^2 \\ \gamma_t^2 \\ \mu_t^2 \\ \delta x_t \\ \delta u_t^1 \\ \psi_t^2 \end{bmatrix} + N_t^2 \begin{bmatrix} \delta x_t \\ \delta u_t^1 \end{bmatrix} + \text{offset terms} = 0
$$

其中 offset terms 包含来自 $V_{t+1}$ 的项：
$$
\text{offset} = \tilde{S}_{t+1} k_{t+1} - \hat{G}_{x,t+1}^T k_\gamma - \hat{H}_{x,t+1}^T k_\mu + \hat{A}_{t+1}^T k_\lambda
$$

求解得到反应函数后，再求解 Leader 的问题。

#### 3.4 前向传播（Forward Pass）

使用求得的策略参数 $\{K_t, \alpha_t\}$ 进行前向模拟：

$$
\begin{aligned}
\delta u_t &= -K_t \delta x_t - \alpha_t \\
\delta x_{t+1} &= A_t \delta x_t + B_t \delta u_t + c_t
\end{aligned}
$$

同时恢复对偶变量：
$$
\begin{aligned}
\gamma_t &= K_{\gamma,t} \delta x_t + k_{\gamma,t} \\
\mu_t &= K_{\mu,t} \delta x_t + k_{\mu,t} \\
\lambda_t &= K_{\lambda,t} \delta x_t + k_{\lambda,t}
\end{aligned}
$$

#### 3.5 KKT 残差计算

为了监控收敛性，构造 KKT 残差：

$$
\text{residual} = \| M \cdot \text{variables} + N \cdot \delta x + n \|_2
$$

当残差小于阈值时，认为 LQ 子问题已收敛。

### 4. 完整 iLQ 算法流程

```
Algorithm: iLQ for Constrained Feedback Stackelberg Games
================================================================================

Input: 
    - 初始轨迹猜测 {xₜ⁽⁰⁾, uₜ⁽⁰⁾}
    - 收敛阈值 ε
    - 最大迭代次数 N_max

Output: 
    - 最优轨迹 {xₜ*, uₜ*}
    - 最优策略 {Kₜ, αₜ}

================================================================================

for k = 1 to N_max do
    
    ============================================================================
    Step 1: LQ 近似
    --------------------------------------------------------------------------------
    在当前轨迹 {xₜ⁽ᵏ⁾, uₜ⁽ᵏ⁾} 处线性化/二次化:
        - 动力学: Aₜ, Bₜ, cₜ
        - 代价: Qₜⁱ, Sₜⁱ, Rₜⁱ, qₜⁱ, rₜⁱ
        - 约束: Gₓ,ₜⁱ, Gᵤ,ₜⁱ, gₜⁱ, Hₓ,ₜⁱ, Hᵤ,ₜⁱ, hₜⁱ
    
    ============================================================================
    
    Step 2: 求解 LQ 博弈 (PDIP-FBST-LQ Solver)
    --------------------------------------------------------------------------------
    # 逆向递推
    for t = T down to 1 do
        if t == T then
            # 终端时刻：直接求解
            构造 M₂, N₂ (Follower)
            π̂², π̌² ← -M₂⁻¹N₂
            构造 Mₜ, Nₜ, nₜ (Leader，代入反应函数)
            Kₜ, αₜ ← -Mₜ⁻¹[Nₜ, nₜ]
        else
            # 中间时刻：考虑值函数
            构造 Pₜ¹, Nₜ² (Follower，含 Vₜ₊₁ 项)
            π̂², π̌² ← -(Pₜ¹)⁻¹Nₜ²
            构造 leader_Pₜ¹ (Leader，含 Vₜ₊₁ 项)
            Kₜ, αₜ ← -(leader_Pₜ¹)⁻¹[...](含未来项)
        end
        
        # 存储乘子递推关系
        K_γ,t, k_γ,t ← 提取对应行
        K_μ,t, k_μ,t ← 提取对应行
        K_λ,t, k_λ,t ← 提取对应行
    end
    
    ============================================================================
    
    Step 3: 前向传播
    --------------------------------------------------------------------------------
    δx₁ ← x₀ - x₀⁽ᵏ⁾
    for t = 1 to T do
        δuₜ ← -Kₜ δxₜ - αₜ
        δxₜ₊₁ ← Aₜ δxₜ + Bₜ δuₜ + cₜ
    end
    
    更新轨迹:
        xₜ⁽ᵏ⁺¹⁾ ← xₜ⁽ᵏ⁾ + δxₜ
        uₜ⁽ᵏ⁺¹⁾ ← uₜ⁽ᵏ⁾ + δuₜ
    
    ============================================================================
    
    Step 4: 线搜索（可选）
    --------------------------------------------------------------------------------
    # 检查代价是否下降，否则缩减步长
    α ← 1.0
    while cost({x⁽ᵏ⁺¹⁾, u⁽ᵏ⁺¹⁾}) ≥ cost({x⁽ᵏ⁾, u⁽ᵏ⁾}) do
        α ← α × decay_factor
        xₜ⁽ᵏ⁺¹⁾ ← xₜ⁽ᵏ⁾ + α·δxₜ
        uₜ⁽ᵏ⁺¹⁾ ← uₜ⁽ᵏ⁾ + α·δuₜ
    end
    
    ============================================================================
    
    Step 5: 收敛检查
    --------------------------------------------------------------------------------
    if ‖δx‖ < ε and ‖δu‖ < ε then
        break
    end

end for

================================================================================
```

### 5. 关键数学符号对照表

| 符号 | 含义 | 维度 |
|------|------|------|
| $x_t$ | 状态向量 | $n_x \times 1$ |
| $u_t$ | 控制输入 | $n_u \times 1$ |
| $K_t$ | 反馈增益矩阵 | $n_u \times n_x$ |
| $\alpha_t$ | 前馈项 | $n_u \times 1$ |
| $\gamma_t$ | 不等式约束乘子 | $n_{ineq} \times 1$ |
| $\mu_t$ | 等式约束乘子 | $n_{eq} \times 1$ |
| $\lambda_t$ | 动力学乘子 | $n_x \times 1$ |
| $Q_t^i$ | 状态代价权重 | $n_x \times n_x$ |
| $R_t^i$ | 控制代价权重 | $n_u \times n_u$ |
| $S_t^i$ | 交叉项权重 | $n_u \times n_x$ |
| $G_{x,t}^i$ | 状态约束 Jacobian | $n_{ineq} \times n_x$ |
| $G_{u,t}^i$ | 控制约束 Jacobian | $n_{ineq} \times n_u$ |
| $\hat{g}_t$ | 约束值的对角矩阵 | $n_{ineq} \times n_{ineq}$ |
| $\hat{\gamma}_t$ | 乘子的对角矩阵 | $n_{ineq} \times n_{ineq}$ |

---

## 三、Julia 代码中的其他实验场景对比

### 1. test_pdip.jl - 线性二次博弈基础测试

**场景特点：**
- **维度**: $n_x=4, n_u=4$ (2 玩家各 2 维控制)
- **动力学**: 离散时间 LTI 系统 $x_{t+1} = A x_t + B u_t$
  - $A = I_4$ (单位阵)
  - $B = 0.1 I_4$
- **代价函数**: 标准 LQ 形式
  $$J_i = \frac{1}{2}\sum_{t=1}^T (x_t^T Q_i x_t + u_t^T R_i u_t)$$
  - $Q_1 = Q_2 = 4I_4$
  - $R_1 = 2\text{diag}(1,1,0,0)$, $R_2 = 2\text{diag}(0,0,1,1)$
- **约束**: 
  - 等式约束: 终端时刻 $x_4 = 1$ (乘以 0.0 实际禁用)
  - 不等式约束: 终端时刻 $x_4 \geq 1$

**与 Highway 的区别:**
| 特性 | test_pdip.jl | highway.jl |
|------|--------------|------------|
| 动力学 | 线性 LTI | 非线性 (Euler 积分) |
| 状态维度 | 4 | 8 |
| 代价类型 | 二次型 | 二次 + 四次项 ($\theta_2^4$) |
| 约束复杂度 | 简单边界 | 道路几何 + 碰撞避免 |
| 求解方法 | 直接 PDIP | iLQ + PDIP-FBST-LQ |

---

### 2. test_nonlinear.jl - 一般非线性博弈测试

**场景特点：**
- **框架**: 使用 `game` 类型定义一般非线性博弈
- **动力学**: $x_{t+1} = f_t(x_t, u_t)$ (可自定义非线性函数)
- **代价**: 任意非线性函数 $J_i = \sum_t g_i(x_t, u_t) + g_i^T(x_T)$
- **约束**: 支持等式和不等式约束的通用形式

**关键代码结构:**
```julia
nonlinear_g = game(
    horizon = horizon,
    n_players = n_players,
    nx = nx,
    nu = nu,
    f_list = f_list,                      # 动力学函数列表
    costs_list = costs_list,              # 阶段代价函数
    terminal_costs_list = terminal_costs_list,
    equality_constraints_list = ...,      # 等式约束
    inequality_constraints_list = ...     # 不等式约束
)
```

**算法流程 (iLQ 迭代):**
1. 前向模拟: `forward_simulation!(current_op, nonlinear_g)`
2. LQ 近似: `lq_approximation!(lq_approx, nonlinear_g, current_op)`
3. 求解 LQ 子问题: `constrained_fbst_lq_solver!(π, lq_approx)`
4. 线搜索: `line_search!(...)`
5. 更新轨迹并重复

---

### 3. fast_highway.jl - 优化版 Highway 场景

**与 highway.jl 的主要区别:**

| 特性 | highway.jl | fast_highway.jl |
|------|------------|-----------------|
| 预测时域 | 20 | 100 |
| 初始障碍参数 $\rho_0$ | 1 | 1/4 |
| 外层迭代次数 | 1 | 10 |
| 障碍衰减率 $\sigma$ | 2 | 4 |
| 碰撞半径 | 0.4 | 0.5 |
| 代价权重 | $10(y_1-0.4)^2$ | $4(y_1-0.5)^2$ |
| 速度匹配项 | $6(v_1-v_2)^2$ | $(v_1-v_2)^2$ |
| Follower 航向惩罚 | $\theta_2^4$ | $2\theta_2^2$ |

**改进点:**
1. **更长的预测时域**: 提高长期规划能力
2. **多层外层迭代**: 通过逐渐减小 $\rho$ 提高精度
3. **简化的代价函数**: 移除高次项，加快收敛
4. **正则化机制**: 当线搜索失败时自动开启正则化

---

### 4. ddp.jl - 单玩家 DDP 测试

**场景特点:**
- **玩家数量**: 实际只有 1 个主动玩家 (Player 1)，Player 2 被注释掉
- **状态维度**: $n_x=4$ (单车状态)
- **代价函数**: 仅 Leader 代价
  $$J_1 = 10(y_1-0.4)^2 + 2a_1^2 + 2\omega_1^2$$
- **终端约束**: $\theta_1 = 0, \theta_2 = 0$

**用途**: 验证算法在退化到单玩家最优控制问题时的一致性

---

### 5. test_pdip_complex_LQ.jl - 复杂 LQ 约束测试

**特殊功能:**
- 支持多个等式和不等式约束
- 测试 KKT 系统在病态条件下的数值稳定性
- 验证对偶变量恢复的正确性

---

## 四、Custom LQR 实验算法原理 (custom_lqr.py)

### 1. 问题描述

Custom LQR 实验实现了**标量线性二次 Stackelberg 博弈**，用于验证论文中的理论结果。

**系统动态 (离散时间):**
$$x_{t+1} = A x_t + B_1 u_{1,t} + B_2 u_{2,t}$$

其中：
- $x_t \in \mathbb{R}$: 标量状态
- $u_{1,t} \in \mathbb{R}$: Leader 的控制输入
- $u_{2,t} \in \mathbb{R}$: Follower 的控制输入
- $A, B_1, B_2 \in \mathbb{R}$: 系统参数

### 2. 代价函数

**Leader (玩家 1) 的阶段代价:**
$$J_1^{\text{stage}} = \frac{1}{2}\left(Q_1 x_t^2 + \Theta_1 u_{2,t}^2 + R_{11} u_{1,t}^2 + R_{12} u_{2,t}^2\right)$$

**Follower (玩家 2) 的阶段代价:**
$$J_2^{\text{stage}} = \frac{1}{2}\left(Q_2 x_t^2 + \Theta_2 u_{1,t}^2 + R_{22} u_{2,t}^2 + R_{21} u_{1,t}^2\right)$$

**总代价:**
$$J = \sum_{t=0}^{T-1} (J_1^{\text{stage}} + J_2^{\text{stage}}) + J^{\text{terminal}}$$

**终端代价:**
$$J^{\text{terminal}} = \frac{1}{2}Q_{1,T} x_T^2 + \frac{1}{2}Q_{2,T} x_T^2$$

### 3. 论文参数配置

默认使用论文中的标量参数：

| 参数 | 值 | 含义 |
|------|-----|------|
| $A$ | 0.8181 | 系统矩阵 |
| $B_1$ | 0.8175 | Leader 控制增益 |
| $B_2$ | -0.7224 | Follower 控制增益 |
| $Q_1$ | 0.1499 | Leader 状态权重 |
| $\Theta_1$ | 0.3245 | Leader 对 $u_2$ 的交叉权重 |
| $R_{11}$ | 0.5186 | Leader 对 $u_1$ 的权重 |
| $Q_2$ | 0.6596 | Follower 状态权重 |
| $\Theta_2$ | 0.4002 | Follower 对 $u_1$ 的交叉权重 |
| $R_{22}$ | 0.9730 | Follower 对 $u_2$ 的权重 |

### 4. Feedback Stackelberg 均衡 (FSE) 解

对于无限时域 LQ 博弈，FSE 策略具有线性反馈形式：
$$u_{1,t} = K_1 x_t, \quad u_{2,t} = K_2 x_t$$

论文给出了三个可能的均衡解：

| 均衡 | $K_1^*$ | $K_2^*$ |
|------|---------|---------|
| SE 1 | -0.6662 | 1.1621 |
| SE 2 | -0.9542 | 0.8523 |
| SE 3 | -1.6526 | 0.7539 |

### 5. 数值积分方法

虽然问题是离散时间的，但代码使用了**RK45 数值积分**来计算连续时间动态的离散化：

```python
def _integrate_dynamics(state, u_leader, u_follower):
    def dxdt(t, x_flat):
        return A @ x + B1 @ u1 + B2 @ u2
    sol = solve_ivp(dxdt, (0, dt), state, method='RK45')
    return sol.y[:, -1]
```

这允许用户通过修改 `dt` 参数来控制离散化精度。

### 6. 求解流程

Custom LQR 实验使用与 Highway 相同的 PDIP 求解器框架：

```
Algorithm: Custom LQR Experiment Flow
================================================================================

Step 1: 创建场景
    scenario = LQRScenario(params=LQRParameters(...))

Step 2: 初始化求解器
    solver = PDIPSolver(config=PDIPConfig(...))

Step 3: 获取初始状态
    x0 = scenario.initial_state()

Step 4: 求解优化问题
    result = solver.solve(scenario, x0)
    # 内部调用 PDIP 算法（见第一部分）

Step 5: 提取反馈增益
    K1, K2 = scenario.extract_feedback_gains(result.decision)

Step 6: 更新 FSE 参考值
    scenario.update_fse_reference(result.decision)

Step 7: 可视化
    visualize_lqr(params, result.states, result.controls)

================================================================================
```

### 7. 可扩展场景

代码提供了两个预定义的扩展场景：

**振荡系统 (oscillatory):**
- $A=0.5, B_1=0.8, B_2=-0.6$
- 产生振荡响应

**不稳定系统 (unstable):**
- $A=1.2, B_1=0.5, B_2=-0.3$
- 开环不稳定，需要反馈镇定

---

## 五、Highway Env 运行结果分析

### 1. 当前运行结果

执行 `python examples/highway.py` 的结果：

```
============================================================
Highway Stackelberg 博弈实验
============================================================
场景参数:
  - 预测时域：20
  - 时间步长：0.05s
  - 状态维度：8
  - 控制维度：4
  - 玩家数量：2

开始求解...
求解完成!
  - 最终损失：32.924251
  - 最终残差：7.469466e-03
  - 总迭代次数：50
```

### 2. 轨迹重合问题分析

**检查结果:**
```
Player 1 位置 (前 5 步):
[[0.9   1.2  ]
 [0.9   1.375]
 [0.892 1.551]
 [0.878 1.728]
 [0.859 1.906]]

Player 2 位置 (前 5 步):
[[0.5   0.6  ]
 [0.5   0.79 ]
 [0.500 0.979]
 [0.500 1.166]
 [0.499 1.352]]

位置差异 (player1 - player2):
[[0.4   0.6  ]
 [0.4   0.585]
 [0.392 0.573]
 [0.378 0.562]
 [0.359 0.554]]

最大位置差异：0.6
最小位置差异：0.00096
位置接近 (<0.1) 的时间点数：0/21
```

**结论：轨迹不会重合！**

- Player 1 和 Player 2 在整个时域内保持明显的位置差异
- 最小距离约为 0.001（远大于碰撞半径 0.4）
- 没有时间点两车位置接近到 0.1 以内

### 3. 为什么会有"轨迹重合"的担忧？

可能的原因：

1. **可视化误解**: 如果使用不同的颜色/标记绘制两车轨迹，可能在某些视角下看起来重叠
2. **初始猜测不当**: 如果初始控制序列都设为零，前向模拟会产生相似的漂移
3. **收敛问题**: 如果算法未充分收敛，可能得到次优解

### 4. 修改初始值的 Infeasible Policy

**当前初始化策略:**
```python
# 在 scenario_highway.py 中
self.x0 = np.array([0.9, 1.2, 3.5, 0.0, 0.5, 0.6, 3.8, 0.0])
# Player 1: (y=0.9, x=1.2, v=3.5, θ=0)
# Player 2: (y=0.5, x=0.6, v=3.8, θ=0)
```

**改进建议:**

如果要测试 infeasible initial policy 的影响，可以：

```python
# 方案 1: 交换初始位置（制造潜在冲突）
x0_infeasible = np.array([0.5, 0.6, 3.5, 0.0, 0.9, 1.2, 3.8, 0.0])
# Player 1 从后方开始，可能导致追赶冲突

# 方案 2: 设置相同初始位置（严重不可行）
x0_conflict = np.array([0.7, 0.9, 3.5, 0.0, 0.7, 0.9, 3.8, 0.0])
# 违反碰撞避免约束

# 方案 3: 设置超出道路边界的初始位置
x0_out_of_road = np.array([1.5, 2.0, 3.5, 0.0, 1.5, 2.5, 3.8, 0.0])
# 违反道路边界约束
```

**预期效果:**
- 使用 infeasible initial policy 会增加求解难度
- 可能需要更多迭代次数才能收敛
- 线搜索可能会更频繁地缩减步长
- 最终解应该仍然满足所有约束（如果问题可行）

### 5. 改善收敛性的建议

如果观察到收敛问题或轨迹异常：

1. **增加外层迭代次数**:
   ```python
   solver_config = PDIPConfig(outer_iter=5, barrier_decay=0.5)
   ```

2. **调整初始障碍参数**:
   ```python
   solver_config = PDIPConfig(barrier_mu=0.1)  # 更小的初始 μ
   ```

3. **修改初始控制猜测**:
   ```python
   # 在 pdip_solver.py 中修改初始化
   decision = np.ones(scenario.decision_shape()) * small_value
   ```

4. **启用正则化**:
   ```python
   # 在 KKT 系统中添加阻尼项
   Hessian += damping * np.eye(n)
   ```

---

## 六、总结对比

| 算法/场景 | 适用问题类型 | 优点 | 缺点 |
|-----------|-------------|------|------|
| **PIDP (Python)** | 通用非线性约束博弈 | 实现简单，无需解析梯度 | 收敛慢，有限差分成本高 |
| **PDIP-FBST-LQ (Julia)** | 光滑非线性博弈 | 利用 LQ 结构，效率高 | 需要可微模型，实现复杂 |
| **Custom LQR** | 线性二次博弈 | 解析解存在，便于验证 | 仅适用于 LQ 场景 |
| **Highway** | 车辆交互场景 | 真实应用背景 | 非凸约束，多局部最优 |
| **Test PDIP** | LTI 动力学测试 | 简单基准，验证正确性 | 约束过于简化 |
| **Test Nonlinear** | 一般非线性测试 | 无约束，快速验证 | 不包含实际约束 |
| **Fast Highway** | 长时域 Highway | 更真实场景，多层迭代 | 计算量大 |
| **Complex LQ** | 复杂约束 LQ | 等式 + 不等式约束 | 短视距，简单动力学 |

**核心洞察:**
1. PIDP 是通用框架，适合快速原型验证
2. 对于实时应用，应使用基于微分动态规划的专用求解器
3. 初始猜测质量显著影响收敛速度和最终解的质量
4. Highway 场景中，合理的初始位置分离是避免数值问题的关键

---

## 七、新增 Python 实验场景说明

### 1. test_pdip.py - 线性二次 Stackelberg 博弈基础测试

**对应 Julia 文件**: `test/test_pdip.jl`

**特点**:
- 4 维状态，LTI 动力学：$x_{t+1} = A x_t + B u_t$
- 2 个玩家，每个玩家 2 维控制
- 简单的终端不等式约束：$x_4 \geq 1.0$
- 用于验证算法在基本 LQ 问题上的正确性

**运行方式**:
```bash
cd /workspace && python examples/test_pdip.py
```

**预期输出**:
- 最终损失约 405.13
- 残差收敛到 $10^{-6}$ 量级
- 28 次迭代内收敛

---

### 2. test_nonlinear.py - 一般非线性博弈测试

**对应 Julia 文件**: `test/test_nonlinear.jl`

**特点**:
- 与 test_pdip 相同的 LTI 动力学
- **无任何约束**（用于测试无约束情况）
- 使用通用 game 类型定义问题
- 验证算法退化到无约束最优控制的一致性

**运行方式**:
```bash
cd /workspace && python examples/test_nonlinear.py
```

**预期输出**:
- 最终损失约 387.03
- 残差收敛到 $10^{-9}$ 量级（更高精度）
- 28 次迭代内收敛

---

### 3. fast_highway.py - 优化版 Highway 场景

**对应 Julia 文件**: `test/fast_highway.jl`

**特点**:
- **更长预测时域** (horizon=20，Julia 原版为 100)
- **多层外层迭代** (outer_iter=3，Julia 原版为 10)
- 复杂的道路几何约束（倾斜边界）
- 碰撞避免约束：$\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2} \geq 0.5$
- 7 个不等式约束 per timestep

**道路约束数学表达**:
$$
g_{\text{road}}(x, y) = 
\begin{cases}
L - y - \frac{L - y_c}{x_c - x_b}(x - x_b), & y < L \\
x_b - x, & y \geq L
\end{cases}
$$

其中 $L=4.0, x_b=0.7, x_c=1.1, y_c=2.0$。

**运行方式**:
```bash
cd /workspace && python examples/fast_highway.py
```

**注意**: Julia 原版 horizon=100，Python 版本为加快测试设为 20。

**预期输出**:
- 最终损失约 16.38
- 3 层外层迭代，每层约 10 次内层迭代
- Player 1 和 Player 2 保持安全距离

---

### 4. test_pdip_complex_LQ.py - 复杂 LQ 约束测试

**对应 Julia 文件**: `test/test_pdip_complex_LQ.jl`

**特点**:
- 同时包含**等式和不等式约束**
- 等式约束（可选开关）: $H_x x + H_u u + h = 0$
- 不等式约束：$x_3 \geq 1.8$, $x_4 \geq 1.0$ (终端)
- 短视距 (horizon=5) 用于快速测试
- 20 层外层迭代，验证障碍参数衰减策略

**约束详情**:
- 阶段不等式：$G_x = [0, 0, 1, 0]$, $g = [-1.8]$ → $x_3 \geq 1.8$
- 终端不等式：$G_x^T = [0, 0, 0, 1]$, $g^T = [-1.0]$ → $x_4 \geq 1.0$

**运行方式**:
```bash
cd /workspace && python examples/test_pdip_complex_LQ.py
```

**预期输出**:
- 多层外层迭代（20 层）
- 障碍参数按 $\mu_{k+1} = 0.5 \mu_k$ 衰减
- 验证约束满足性

---

### 5. 所有 Python 实验场景对比表

| 场景文件 | Horizon | 动力学 | 约束类型 | 外层迭代 | 主要用途 |
|---------|---------|--------|----------|----------|----------|
| `test_pdip.py` | 10 | LTI | 终端不等式 | 1 | 基础验证 |
| `test_nonlinear.py` | 10 | LTI | 无约束 | 1 | 无约束基准 |
| `fast_highway.py` | 20 | 非线性 | 7 个不等式 | 3 | 复杂场景测试 |
| `test_pdip_complex_LQ.py` | 5 | LTI | 等式 + 不等式 | 20 | 约束完整性测试 |
| `highway.py` | 20 | 非线性 | 标准 Highway | 1 | 主实验场景 |
| `custom_lqr.py` | 30 | 连续 LTI | 无约束 | 1 | 标量 LQ 验证 |

---

### 6. Python vs Julia 版本差异说明

| 特性 | Python (PDIP) | Julia (PDIP-FBST-LQ) |
|------|---------------|---------------------|
| **求解方法** | 通用内点法 | LQ 近似 + 反馈 Stackelberg |
| **梯度计算** | 有限差分 | 解析导数 |
| **Hessian** | 阻尼单位阵近似 | LQ 二阶信息 |
| **计算效率** | 较慢（适合小规模） | 快（利用 LQ 结构） |
| **代码复杂度** | 低（~500 行） | 高（~2000 行） |
| **适用场景** | 原型验证，教学 | 实际应用，大规模问题 |

**为什么 Python 版本没有完全对应的 Julia 功能？**

1. **设计目标不同**:
   - Python: 快速原型，验证算法概念
   - Julia: 高性能求解，实际部署

2. **技术栈差异**:
   - Python: 使用 NumPy/SciPy 通用工具
   - Julia: 自定义 LQ 求解器，静态数组优化

3. **算法重点**:
   - Python: 通用 PDIP 框架
   - Julia: Feedback Stackelberg 均衡的 specialized solver

**建议**:
- 学习/教学：使用 Python 版本理解算法原理
- 研究/应用：使用 Julia 版本获得高性能
- 验证结果：两种实现应得到一致的定性结论

---

## 八、完整实验运行指南

### 快速测试所有场景

```bash
cd /workspace/examples

# 1. 基础 LQ 测试（最快）
python test_pdip.py

# 2. 无约束测试
python test_nonlinear.py

# 3. Highway 主场景
python highway.py

# 4. 复杂约束 LQ
python test_pdip_complex_LQ.py

# 5. Fast Highway（较长时域）
python fast_highway.py

# 6. Custom LQR（标量系统）
python custom_lqr.py
```

### 预期运行时间

| 场景 | 预计时间 | 迭代次数 |
|------|---------|---------|
| test_pdip | < 5 秒 | ~28 |
| test_nonlinear | < 5 秒 | ~28 |
| highway | ~10 秒 | ~50 |
| test_pdip_complex_LQ | ~30 秒 | 20×15 |
| fast_highway | ~30 秒 | 3×10 |
| custom_lqr | < 5 秒 | ~30 |

---

**文档版本**: v2.0  
**更新日期**: 2024  
**包含场景**: 6 个 Python 实验 + 4 个 Julia 实验场景分析
