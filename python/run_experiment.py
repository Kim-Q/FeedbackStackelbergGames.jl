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
        ],
    )
    parser.add_argument("--input", help="Input .npz file for data processing or plotting.")
    parser.add_argument("--output-dir", default="python_outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--outer-iter", type=int, default=1)
    parser.add_argument("--barrier-mu", type=float, default=1.0)
    parser.add_argument("--barrier-decay", type=float, default=0.5)
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
    runner.run(args.env, input_path=args.input)


if __name__ == "__main__":
    main()
