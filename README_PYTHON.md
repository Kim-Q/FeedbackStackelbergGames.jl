# Python Stackelberg 博弈实验

本项目实现了基于 **PDIP（原始 - 对偶内点法）**的 Stackelberg 博弈求解器，支持 Highway 和自定义 LQR 两种实验场景。

## 目录结构

```
/workspace
├── python/
│   └── feedback_stackelberg/
│       ├── __init__.py          # 包初始化
│       ├── config.py            # 配置参数类
│       ├── pdip_solver.py       # PDIP 求解器（核心算法）
│       ├── scenario_highway.py  # Highway 场景定义
│       ├── scenario_lqr.py      # LQR 场景定义
│       ├── visualization.py     # 可视化模块
│       └── io_utils.py          # 输入输出工具
├── examples/
│   ├── highway.py               # Highway 实验示例
│   └── custom_lqr.py            # 自定义 LQR 实验示例
├── requirements.txt             # Python 依赖
└── README_PYTHON.md            # 本文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖包：
- numpy >= 1.26.4
- matplotlib >= 3.8.4
- pillow >= 12.2.0

## 快速开始

### 1. Highway 实验

运行 Highway 场景下的 Stackelberg 博弈实验：

```bash
cd /workspace
python examples/highway.py
```

**参数选项：**
```bash
python examples/highway.py \
    --output-dir python_outputs \
    --max-iter 50 \
    --outer-iter 1 \
    --barrier-mu 1.0 \
    --barrier-decay 0.5 \
    --residual-tol 1e-4
```

**输出文件：**
- `python_outputs/highway_*/highway_*.csv`: 状态轨迹、控制输入、损失历史、残差历史
- `python_outputs/highway_*/highway_*.png`: 轨迹可视化图
- `python_outputs/highway_*/highway_*.gif`: 动画

### 2. 自定义 LQR 实验

运行自定义 LQR 场景：

```bash
python examples/custom_lqr.py
```

LQR 环境使用反馈增益作为优化变量：u₁ = K₁ x，u₂ = K₂ x，收敛后的 K₁/K₂ 会写入输出元数据并作为 FSE 参考值。

**选择不同的场景：**
```bash
# 默认场景
python examples/custom_lqr.py --scenario default

# 振荡系统（二阶谐振子）
python examples/custom_lqr.py --scenario oscillatory

# 不稳定系统（倒立摆线性化模型）
python examples/custom_lqr.py --scenario unstable
```

## 核心算法：PDIP（原始 - 对偶内点法）

### 算法原理

PDIP 用于求解带不等式约束的优化问题：

```
min f(u)
s.t. g(u) >= 0
```

通过引入松弛变量 s > 0，将不等式转为等式：`g(u) - s = 0`

构造障碍问题：`min f(u) - μ * Σlog(s_i)`

### 主要计算步骤

在 `pdip_solver.py` 的 `solve()` 方法中，算法分为以下步骤：

```python
# ========== 步骤 1: 初始化原始变量和对偶变量 ==========
x0 = initial_state
decision = zeros(...)
slack = ones(...) * initial_slack
dual = ones(...) * initial_dual
mu = barrier_mu

# ========== 步骤 2: 外层循环 - 更新障碍参数 ==========
for outer_iter in range(outer_iter):
    
    # ========== 步骤 3: 内层牛顿迭代 ==========
    for iteration in range(max_iter):
        
        # ----- 步骤 3.1: 前向模拟得到状态轨迹 -----
    states, controls = rollout(x0, decision)
        
        # ----- 步骤 3.2: 计算约束与目标函数 -----
    constraints = collect_constraints(states, controls)
    total_cost = total_cost(states, controls)
        
        # ----- 步骤 3.3: 构造 KKT 残差 -----
        # KKT 条件：
        # 1. 平稳性：∇f(u) - J(u)^T * λ = 0
    # 2. 可行性：g(u) - s = 0
        # 3. 互补松弛：s_i * λ_i = μ
        grad = finite_difference_gradient(...)
        jacobian = finite_difference_jacobian(...)
        r_dual, r_pri, r_cent = kkt_residuals(...)
        
        # ----- 步骤 3.4: 求解牛顿方向 -----
        # 求解 KKT 线性系统
        delta_u, delta_dual, delta_slack = solve_kkt_system(...)
        
        # ----- 步骤 3.5: 线搜索保证正性约束 -----
        step = line_search(slack, dual, delta_slack, delta_dual)
        
        # ----- 步骤 3.6: 更新原始变量与对偶变量 -----
    decision += step * delta_u
        slack += step * delta_slack
        dual += step * delta_dual
    
    # 衰减障碍参数
    mu *= barrier_decay
```

### 关键公式

**KKT 系统：**
```
[ H   -J^T   0  ] [Δu  ]   [-r_dual ]
[ J    0    -I  ] [Δλ  ] = [-r_pri  ]
[ 0    S    Λ  ] [Δs  ]   [-r_cent ]
```

其中：
- H = δ*I（阻尼单位阵近似 Hessian）
- J = ∂g/∂u（约束雅可比矩阵）
- S = diag(s)（松弛变量对角阵）
- Λ = diag(λ)（对偶变量对角阵）

## 扩展到新环境

### 创建自定义场景

要实现新的实验环境，需要创建一个场景类，实现以下接口：

```python
class CustomScenario:
    def __init__(self, params):
        self.params = params
        self.nx = ...  # 状态维度
        self.nu = ...  # 控制维度
        self.players_u_index_list = (...)  # 玩家控制索引
    
    def initial_state(self) -> np.ndarray:
        """返回初始状态"""
        pass
    
    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """系统动力学"""
        pass
    
    def stage_cost(self, state: np.ndarray, control: np.ndarray) -> float:
        """阶段成本"""
        pass
    
    def terminal_cost(self, state: np.ndarray) -> float:
        """终端成本"""
        pass
    
    def forward_simulation(self, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """前向仿真"""
        pass

    def decision_shape(self) -> tuple[int, ...]:
        """决策变量形状（默认控制序列，LQR 可返回反馈增益矩阵形状）"""
        pass

    def rollout(self, x0: np.ndarray, decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """根据决策变量生成状态与控制序列"""
        pass
    
    def total_cost(self, states: np.ndarray, controls: np.ndarray) -> float:
        """总成本"""
        pass
    
    def collect_constraints(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """收集所有约束（不等式约束 g(u) >= 0）"""
        pass
```

### 使用示例

```python
from feedback_stackelberg.pdip_solver import PDIPSolver
from feedback_stackelberg.config import PDIPConfig

# 创建场景
scenario = CustomScenario(params)

# 创建求解器
solver_config = PDIPConfig(max_iter=50, residual_tol=1e-6)
solver = PDIPSolver(config=solver_config)

# 求解
result = solver.solve(scenario, initial_state)

# 获取结果
states = result.states
controls = result.controls
loss_history = result.loss_history
```

## 代码风格说明

- 使用 class 封装，避免函数嵌套
- 每个方法都有清晰的 docstring 注释
- 算法关键步骤使用中文注释标注
- 支持在不同实验环境间切换

## 参考图像

生成的可视化图像参考 `src/` 目录中的模板：
- `highway_st.png`: Highway 场景轨迹图
- `fbst_fbne.png`: 对比图
- 其他 GIF 动画

## 常见问题

**Q: 如何调整收敛速度？**
A: 增加 `max_iter` 或减小 `residual_tol`，也可以调整 `barrier_decay` 来控制障碍参数的衰减速率。

**Q: 如何处理无约束问题？**
A: 在场景的 `collect_constraints()` 方法中返回空数组即可。

**Q: 如何保存中间结果？**
A: `result.loss_history` 和 `result.residual_history` 记录了每次迭代的收敛过程。
