#!/usr/bin/env python3
"""
自定义 LQR 实验示例

演示如何在自定义的 LQR 环境中使用 PDIP 求解器。
用户可以轻松切换不同的 LQR 参数配置。

使用方法:
    python examples/custom_lqr.py
    
输出:
    - python_outputs/lqr_*.csv: 状态和控制数据
    - python_outputs/lqr_*.png: 可视化图
"""

import sys
from pathlib import Path

# 添加 python 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np
from feedback_stackelberg.config import PDIPConfig
from feedback_stackelberg.pdip_solver import PDIPSolver
from feedback_stackelberg.scenario_lqr import LQRParameters, LQRScenario
from feedback_stackelberg.visualization import visualize_lqr
from feedback_stackelberg.io_utils import ExperimentIO, ExperimentOutput


class CustomLQRExperiment:
    """
    自定义 LQR 实验类
    
    封装了 LQR 场景的配置、求解和结果保存流程。
    支持快速切换不同的 LQR 参数配置。
    """
    
    def __init__(
        self,
        output_dir: str = "python_outputs",
        lqr_params: LQRParameters | None = None,
        solver_config: PDIPConfig | None = None,
    ):
        """
        初始化实验
        
        Args:
            output_dir: 输出目录
            lqr_params: LQR 参数配置，若为 None 则使用默认配置
            solver_config: PDIP 求解器配置，若为 None 则使用默认配置
        """
        self.output_dir = output_dir
        self.lqr_params = lqr_params or self._default_lqr_params()
        self.solver_config = solver_config or PDIPConfig(
            max_iter=50,
            outer_iter=1,
            barrier_mu=1.0,
            barrier_decay=0.5,
            residual_tol=1e-6,
        )
        
        # 创建场景和求解器
        self.scenario = LQRScenario(params=self.lqr_params)
        self.solver = PDIPSolver(config=self.solver_config)
    
    def _default_lqr_params(self) -> LQRParameters:
        """默认 LQR 参数配置"""
        return LQRParameters(
            horizon=30,
            dt=0.05,
            dynamics="linear",
            # 系统矩阵 A (2x2)
            A=np.array([[0.0, 1.0], [-1.0, -0.1]], dtype=float),
            # Leader 控制矩阵 B1 (2x1)
            B_leader=np.array([[0.0], [1.0]], dtype=float),
            # Follower 控制矩阵 B2 (2x1)
            B_follower=np.array([[0.0], [0.5]], dtype=float),
            # 状态权重矩阵
            Q_leader=np.diag([1.0, 0.2]).astype(float),
            Q_follower=np.diag([0.8, 0.5]).astype(float),
            # 控制权重矩阵
            R_leader=np.array([[0.5]], dtype=float),
            R_follower=np.array([[0.3]], dtype=float),
            # 交叉耦合项
            R_leader_follower=np.array([[0.1]], dtype=float),
            R_follower_leader=np.array([[0.1]], dtype=float),
            # 耦合项
            Theta_leader=np.array([[0.05]], dtype=float),
            Theta_follower=np.array([[0.05]], dtype=float),
            # 初始状态
            x0=np.array([1.0, 0.0], dtype=float),
        )
    
    def run(self, save_output: bool = True) -> ExperimentOutput:
        """
        运行实验
        
        Args:
            save_output: 是否保存输出
            
        Returns:
            ExperimentOutput: 实验结果
        """
        print("=" * 60)
        print("自定义 LQR Stackelberg 博弈实验")
        print("=" * 60)
        print(f"场景参数:")
        print(f"  - 预测时域：{self.lqr_params.horizon}")
        print(f"  - 时间步长：{self.lqr_params.dt}s")
        print(f"  - 动力学类型：{self.lqr_params.dynamics}")
        print(f"  - 状态维度：{self.scenario.nx}")
        print(f"  - 控制维度：{self.scenario.nu}")
        print()
        
        # ========== 步骤 1: 获取初始状态 ==========
        initial_state = self.scenario.initial_state()
        
        # ========== 步骤 2: 使用 PDIP 求解器求解 ==========
        print("开始求解...")
        result = self.solver.solve(self.scenario, initial_state)
        
        print(f"求解完成!")
        print(f"  - 最终损失：{result.loss_history[-1][-1]:.6f}")
        print(f"  - 最终残差：{result.residual_history[-1][-1]:.6e}")
        print(f"  - 总迭代次数：{sum(len(h) for h in result.loss_history)}")
        print()
        
        # ========== 步骤 3: 打包实验结果 ==========
        metadata = {
            "environment": "custom_lqr",
            "dynamics": self.lqr_params.dynamics,
            "horizon": self.lqr_params.horizon,
            "state_dim": self.scenario.nx,
            "control_dim": self.scenario.nu,
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
        
        # ========== 步骤 4: 保存结果和可视化 ==========
        if save_output:
            timestamp = np.datetime_as_string(np.datetime64("now"), unit="s").replace(":", "-")
            filename = f"lqr_{timestamp}"
            
            io = ExperimentIO(self.output_dir)
            io.save(output, filename)
            
            base_name = Path(filename).stem
            vis_files = visualize_lqr(
                self.lqr_params,
                output.states,
                output.controls,
                self.output_dir,
                base_name,
            )
            
            print("输出文件已保存:")
            print(f"  - 数据目录：{self.output_dir}/")
            print(f"  - 轨迹图：{vis_files['png']}")
            print(f"  - 动画：{vis_files['gif']}")
            print()
        
        return output


def create_oscillatory_lqr() -> LQRParameters:
    """创建一个振荡系统的 LQR 参数配置"""
    return LQRParameters(
        horizon=40,
        dt=0.05,
        dynamics="linear",
        # 振荡系统：二阶谐振子
        A=np.array([[0.0, 1.0], [-4.0, -0.2]], dtype=float),
        B_leader=np.array([[0.0], [1.0]], dtype=float),
        B_follower=np.array([[0.0], [0.5]], dtype=float),
        Q_leader=np.diag([2.0, 0.5]).astype(float),
        Q_follower=np.diag([1.5, 0.3]).astype(float),
        R_leader=np.array([[0.3]], dtype=float),
        R_follower=np.array([[0.2]], dtype=float),
        x0=np.array([0.5, 0.0], dtype=float),
    )


def create_unstable_lqr() -> LQRParameters:
    """创建一个不稳定系统的 LQR 参数配置"""
    return LQRParameters(
        horizon=30,
        dt=0.05,
        dynamics="linear",
        # 不稳定系统（倒立摆线性化模型）
        A=np.array([[0.0, 1.0], [9.8, -0.1]], dtype=float),
        B_leader=np.array([[0.0], [1.0]], dtype=float),
        B_follower=np.array([[0.0], [0.5]], dtype=float),
        Q_leader=np.diag([10.0, 1.0]).astype(float),
        Q_follower=np.diag([5.0, 0.5]).astype(float),
        R_leader=np.array([[0.1]], dtype=float),
        R_follower=np.array([[0.1]], dtype=float),
        x0=np.array([0.1, 0.0], dtype=float),
    )


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自定义 LQR Stackelberg 博弈实验")
    parser.add_argument("--output-dir", type=str, default="python_outputs",
                        help="输出目录 (默认：python_outputs)")
    parser.add_argument("--max-iter", type=int, default=50,
                        help="最大迭代次数 (默认：50)")
    parser.add_argument("--scenario", type=str, default="default",
                        choices=["default", "oscillatory", "unstable"],
                        help="选择 LQR 场景 (默认：default)")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存输出文件")
    
    args = parser.parse_args()
    
    # 根据选择创建不同的 LQR 参数配置
    if args.scenario == "oscillatory":
        lqr_params = create_oscillatory_lqr()
    elif args.scenario == "unstable":
        lqr_params = create_unstable_lqr()
    else:
        lqr_params = None  # 使用默认配置
    
    # 创建求解器配置
    solver_config = PDIPConfig(
        max_iter=args.max_iter,
        outer_iter=1,
        barrier_mu=1.0,
        barrier_decay=0.5,
        residual_tol=1e-6,
    )
    
    # 创建并运行实验
    experiment = CustomLQRExperiment(
        output_dir=args.output_dir,
        lqr_params=lqr_params,
        solver_config=solver_config,
    )
    
    experiment.run(save_output=not args.no_save)


if __name__ == "__main__":
    main()
