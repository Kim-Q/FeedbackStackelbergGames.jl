#!/usr/bin/env python3
"""
Highway 实验示例

复现高速公路场景下的 Stackelberg 博弈实验，使用 PDIP（原始 - 对偶内点法）求解。
保存实验结果（状态轨迹、控制输入、损失历史、残差历史）和可视化图像。

使用方法:
    python examples/highway.py
    
输出:
    - python_outputs/highway_*.csv: 状态和控制数据
    - python_outputs/highway_*.png: 轨迹可视化图
    - python_outputs/highway_*.gif: 动画
"""

import sys
from pathlib import Path

# 添加 python 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from feedback_stackelberg.config import ExperimentConfig, PDIPConfig
from feedback_stackelberg.pdip_solver import PDIPSolver
from feedback_stackelberg.scenario_highway import HighwayParameters, HighwayScenario
from feedback_stackelberg.visualization import visualize_highway
from feedback_stackelberg.io_utils import ExperimentIO, ExperimentOutput


def run_highway_experiment(
    output_dir: str = "python_outputs",
    max_iter: int = 50,
    outer_iter: int = 1,
    barrier_mu: float = 1.0,
    barrier_decay: float = 0.5,
    residual_tol: float = 1e-4,
    save_output: bool = True,
) -> ExperimentOutput:
    """
    运行 Highway 实验
    
    Args:
        output_dir: 输出目录
        max_iter: 最大迭代次数
        outer_iter: 外层迭代次数（障碍参数更新次数）
        barrier_mu: 初始障碍参数
        barrier_decay: 障碍参数衰减率
        residual_tol: 收敛容差
        save_output: 是否保存输出
        
    Returns:
        ExperimentOutput: 实验结果
    """
    # ========== 步骤 1: 配置实验参数 ==========
    experiment_config = ExperimentConfig(output_dir=output_dir)
    solver_config = PDIPConfig(
        max_iter=max_iter,
        outer_iter=outer_iter,
        barrier_mu=barrier_mu,
        barrier_decay=barrier_decay,
        residual_tol=residual_tol,
    )
    
    # ========== 步骤 2: 创建 Highway 场景 ==========
    # Highway 场景参数
    highway_params = HighwayParameters(
        horizon=20,      # 预测时域
        dt=0.05,         # 时间步长
        collision_radius=0.4,  # 碰撞半径
    )
    
    # 创建场景（默认角色：player1 为 leader，player2 为 follower）
    scenario = HighwayScenario(params=highway_params, swap_roles=False)
    
    print("=" * 60)
    print("Highway Stackelberg 博弈实验")
    print("=" * 60)
    print(f"场景参数:")
    print(f"  - 预测时域：{scenario.params.horizon}")
    print(f"  - 时间步长：{scenario.params.dt}s")
    print(f"  - 状态维度：{scenario.nx}")
    print(f"  - 控制维度：{scenario.nu}")
    print(f"  - 玩家数量：{scenario.n_players}")
    print()
    
    # ========== 步骤 3: 初始化 PDIP 求解器 ==========
    solver = PDIPSolver(config=solver_config)
    
    # ========== 步骤 4: 求解优化问题 ==========
    print("开始求解...")
    initial_state = scenario.initial_state()
    result = solver.solve(scenario, initial_state)
    
    print(f"求解完成!")
    print(f"  - 最终损失：{result.loss_history[-1][-1]:.6f}")
    print(f"  - 最终残差：{result.residual_history[-1][-1]:.6e}")
    print(f"  - 总迭代次数：{sum(len(h) for h in result.loss_history)}")
    print()
    
    # ========== 步骤 5: 打包实验结果 ==========
    metadata = {
        "environment": "highway",
        "swap_roles": False,
        "horizon": scenario.params.horizon,
        "max_iter": max_iter,
        "final_loss": result.loss_history[-1][-1],
        "final_residual": result.residual_history[-1][-1],
    }
    
    output = ExperimentOutput(
        states=result.states,
        controls=result.controls,
        loss_history=result.loss_history,
        residual_history=result.residual_history,
        metadata=metadata,
    )
    
    # ========== 步骤 6: 保存结果和可视化 ==========
    if save_output:
        import numpy as np
        timestamp = np.datetime_as_string(np.datetime64("now"), unit="s").replace(":", "-")
        filename = f"highway_{timestamp}"
        
        io = ExperimentIO(output_dir)
        io.save(output, filename)
        
        base_name = Path(filename).stem
        vis_files = visualize_highway(scenario, output.states, output_dir, base_name)
        
        print("输出文件已保存:")
        print(f"  - 数据目录：{output_dir}/")
        print(f"  - 轨迹图：{vis_files['png']}")
        print(f"  - 动画：{vis_files['gif']}")
        print()
    
    return output


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Highway Stackelberg 博弈实验")
    parser.add_argument("--output-dir", type=str, default="python_outputs",
                        help="输出目录 (默认：python_outputs)")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="最大迭代次数 (默认：50)")
    parser.add_argument("--outer-iter", type=int, default=1,
                        help="外层迭代次数 (默认：1)")
    parser.add_argument("--barrier-mu", type=float, default=1.0,
                        help="初始障碍参数 (默认：1.0)")
    parser.add_argument("--barrier-decay", type=float, default=0.5,
                        help="障碍参数衰减率 (默认：0.5)")
    parser.add_argument("--residual-tol", type=float, default=1e-4,
                        help="收敛容差 (默认：1e-4)")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存输出文件")
    
    args = parser.parse_args()
    
    run_highway_experiment(
        output_dir=args.output_dir,
        max_iter=args.max_iter,
        outer_iter=args.outer_iter,
        barrier_mu=args.barrier_mu,
        barrier_decay=args.barrier_decay,
        residual_tol=args.residual_tol,
        save_output=not args.no_save,
    )


if __name__ == "__main__":
    main()
