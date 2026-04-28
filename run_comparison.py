#!/usr/bin/env python3
"""
对比运行脚本：
1. 验证 LQR 环境下的策略收敛性 (K1*, K2*)
2. 运行 Highway 环境 (复现 Julia 结果)
3. 所有输出自动保存到带时间戳的子目录
"""

import os
import sys
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python.feedback_stackelberg.config import PDIPConfig, LQRParams, HighwayParams
from python.feedback_stackelberg.scenario_lqr import LQRScenario, FSE_solutions
from python.feedback_stackelberg.scenario_highway import HighwayScenario
from python.feedback_stackelberg.solvers.pdip_solver import NWPDIPFBSTSolver
from python.feedback_stackelberg.utils.io_utils import setup_output_directory

def run_lqr_verification(output_dir):
    """
    运行 LQR 场景并验证策略收敛性
    """
    print("\n" + "="*60)
    print("开始 LQR 场景验证 (验证策略 K* 收敛)")
    print("="*60)
    
    # 配置参数
    config = PDIPConfig(
        max_iter=20,
        barrier_mu=1.0,
        barrier_decay=0.5,
        tol=1e-6,
        line_search_alpha=0.99
    )
    
    # 初始化场景
    scenario = LQRScenario()
    
    # 选择要验证的均衡解 (默认验证 SE 1)
    target_se = "SE 1"
    K1_target = FSE_solutions[target_se]["K1_star"]
    K2_target = FSE_solutions[target_se]["K2_star"]
    
    print(f"目标均衡解: {target_se}")
    print(f"理论最优 K1*: {K1_target:.6f}, K2*: {K2_target:.6f}")
    
    # 初始化求解器
    solver = NWPDIPFBSTSolver(scenario, config)
    
    # 设置初始状态
    x0 = np.array([1.0])  # 初始状态
    
    # 运行求解 (修改 solve 方法以支持打印 K*)
    # 注意：这里我们需要调用带有详细日志的 solve 方法
    # 为了演示，我们手动执行几步迭代逻辑或直接调用增强版 solve
    
    try:
        # 假设 solver 已经更新了 print 逻辑
        results = solver.solve(x0=x0, verbose=True, target_K=(K1_target, K2_target))
        
        # 保存结果
        result_file = os.path.join(output_dir, "lqr_results.npz")
        np.savez(result_file, **results)
        print(f"\nLQR 结果已保存至: {result_file}")
        
        return results
        
    except Exception as e:
        print(f"LQR 求解过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_highway_scenario(output_dir):
    """
    运行 Highway 场景 (复现 Julia 版本结果)
    """
    print("\n" + "="*60)
    print("开始 Highway 场景仿真 (复现 Julia 结果)")
    print("="*60)
    
    # 配置参数 (需与 Julia 版本严格对齐)
    config = PDIPConfig(
        max_iter=20,
        barrier_mu=1.0,
        barrier_decay=0.5,
        tol=1e-6,
        line_search_alpha=0.99
    )
    
    # 初始化 Highway 场景
    # 注意：需要确保 scenario_highway.py 中存在 HighwayScenario 类
    try:
        scenario = HighwayScenario()
    except Exception as e:
        print(f"警告: Highway 场景尚未完全实现，跳过此步骤。错误: {e}")
        return None
    
    solver = NWPDIPFBSTSolver(scenario, config)
    
    # Highway 初始状态 [pos_leader, vel_leader, pos_follower, vel_follower]
    # 示例初始条件
    x0 = np.array([0.0, 20.0, -50.0, 25.0]) 
    
    try:
        results = solver.solve(x0=x0, verbose=True)
        
        # 保存结果
        result_file = os.path.join(output_dir, "highway_results.npz")
        np.savez(result_file, **results)
        print(f"\nHighway 结果已保存至: {result_file}")
        
        return results
    except Exception as e:
        print(f"Highway 求解过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 1. 设置输出目录 (带时间戳)
    output_dir = setup_output_directory(prefix="comparison_run")
    print(f"本次运行结果将保存至: {output_dir}")
    
    # 保存配置文件到输出目录
    config_log = os.path.join(output_dir, "config_log.txt")
    with open(config_log, 'w') as f:
        f.write("Configuration Log\n")
        f.write("=================\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Output Dir: {output_dir}\n")
    
    # 2. 运行 LQR 验证
    lqr_res = run_lqr_verification(output_dir)
    
    # 3. 运行 Highway 验证
    highway_res = run_highway_scenario(output_dir)
    
    print("\n" + "="*60)
    print("所有任务完成")
    print(f"请查看目录: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
