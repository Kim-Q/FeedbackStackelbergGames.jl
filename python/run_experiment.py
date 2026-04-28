import argparse

from feedback_stackelberg.config import ExperimentConfig, PDIPConfig
from feedback_stackelberg.experiments import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stackelberg game experiments (Python).")
    parser.add_argument(
        "--env",
        required=True,
        choices=[
            "highway",
            "highway_without_saving_data",
            "highway_switch_leader_and_follower",
            "highway_multi_run",
            "highway_data_processing",
            "highway_multi_run_plot",
            "lqr",
        ],
    )
    parser.add_argument(
        "--input",
        help="Input CSV/metadata path prefix for data processing or plotting (e.g., highway_*_metadata.json).",
    )
    parser.add_argument("--output-dir", default="python_outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--outer-iter", type=int, default=1)
    parser.add_argument("--barrier-mu", type=float, default=1.0)
    parser.add_argument("--barrier-decay", type=float, default=0.5)
    parser.add_argument("--lqr-config", help="Path to LQR parameter file (.npz or .json).")
    parser.add_argument(
        "--lqr-dynamics",
        choices=["linear", "nonlinear"],
        default="linear",
        help="Choose linear or nonlinear dynamics for the LQR environment.",
    )
    parser.add_argument("--lqr-horizon", type=int, help="Override LQR horizon.")
    parser.add_argument("--lqr-dt", type=float, help="Override LQR time step.")
    args = parser.parse_args()

    experiment_config = ExperimentConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        num_samples=args.num_samples,
    )
    solver_config = PDIPConfig(
        max_iter=args.max_iter,
        outer_iter=args.outer_iter,
        barrier_mu=args.barrier_mu,
        barrier_decay=args.barrier_decay,
    )
    runner = ExperimentRunner(experiment_config, solver_config)
    runner.run(
        args.env,
        input_path=args.input,
        lqr_params_path=args.lqr_config,
        lqr_dynamics=args.lqr_dynamics,
        lqr_horizon=args.lqr_horizon,
        lqr_dt=args.lqr_dt,
    )


if __name__ == "__main__":
    main()
