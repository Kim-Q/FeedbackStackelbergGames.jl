#!/usr/bin/env python3
"""
自定义 LQR 实验示例 - 使用论文中的标量参数

演示如何在自定义的 LQR 环境中使用 PDIP 求解器。
默认使用论文中提供的标量 LQR 参数配置。

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
            lqr_params: LQR 参数配置，若为 None 则使用默认配置（论文参数）
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
        """
        默认 LQR 参数配置（来自论文）
        
        系统动态参数 (标量):
            A = 0.8181      # 系统矩阵
            B1 = 0.8175     # 玩家 1 控制输入矩阵
            B2 = -0.7224    # 玩家 2 控制输入矩阵
        
        玩家 1 的代价函数参数:
            Q1 = 0.1499     # 状态权重
            Theta1 = 0.3245 # 对 u2 的交叉权重
            R11 = 0.5186    # 对 u1 的权重
            R12 = 0         # 对 u2 的权重
        
        玩家 2 的代价函数参数:
            Q2 = 0.6596     # 状态权重
            Theta2 = 0.4002 # 对 u1 的交叉权重
            R22 = 0.9730    # 对 u2 的权重
            R21 = 0         # 对 u1 的权重
        
        FSE 解参考值:
            SE 1: K1=-0.6662, K2=1.1621
            SE 2: K1=-0.9542, K2=0.8523
            SE 3: K1=-1.6526, K2=0.7539
        """
        return LQRParameters(
            horizon=30,
            dt=0.01,  # 积分时间步长
            # 系统动态参数 (标量)
            A=0.8181,
            B1=0.8175,
            B2=-0.7224,
            # 玩家 1 的代价函数参数
            Q1=0.1499,
            Theta1=0.3245,
            R11=0.5186,
            R12=0.0,
            # 玩家 2 的代价函数参数
            Q2=0.6596,
            Theta2=0.4002,
            R22=0.9730,
            R21=0.0,
            # 初始状态
            x0=1.0,
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
        print(f"  - 状态维度：{self.scenario.nx}")
        print(f"  - 控制维度：{self.scenario.nu}")
        print()
        print("系统参数:")
        print(f"  - A={self.lqr_params.A[0,0]:.4f}, B1={self.lqr_params.B1[0,0]:.4f}, B2={self.lqr_params.B2[0,0]:.4f}")
        print(f"  - Q1={self.lqr_params.Q1[0,0]:.4f}, Theta1={self.lqr_params.Theta1[0,0]:.4f}, R11={self.lqr_params.R11[0,0]:.4f}")
        print(f"  - Q2={self.lqr_params.Q2[0,0]:.4f}, Theta2={self.lqr_params.Theta2[0,0]:.4f}, R22={self.lqr_params.R22[0,0]:.4f}")
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
            print(f"  - 动画：{vis_files.get('gif', 'N/A')}")
            print()
        
        return output


def create_oscillatory_lqr() -> LQRParameters:
    """创建一个振荡系统的 LQR 参数配置"""
    return LQRParameters(
        horizon=40,
        dt=0.01,
        # 振荡系统
        A=0.5,
        B1=0.8,
        B2=-0.6,
        Q1=1.0,
        Theta1=0.2,
        R11=0.3,
        R12=0.0,
        Q2=0.8,
        Theta2=0.15,
        R22=0.25,
        R21=0.0,
        x0=0.5,
    )


def create_unstable_lqr() -> LQRParameters:
    """创建一个不稳定系统的 LQR 参数配置"""
    return LQRParameters(
        horizon=30,
        dt=0.01,
        # 不稳定系统
        A=1.2,
        B1=0.5,
        B2=-0.3,
        Q1=2.0,
        Theta1=0.1,
        R11=0.1,
        R12=0.0,
        Q2=1.5,
        Theta2=0.08,
        R22=0.15,
        R21=0.0,
        x0=0.1,
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
        lqr_params = None  # 使用默认配置（论文参数）
    
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
