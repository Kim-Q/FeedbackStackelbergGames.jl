from dataclasses import dataclass


@dataclass
class PDIPConfig:
    max_iter: int = 10
    outer_iter: int = 1
    barrier_mu: float = 1.0
    barrier_decay: float = 0.5
    step_size: float = 1.0
    step_decay: float = 0.5
    line_search_max_iter: int = 20
    residual_tol: float = 1e-4
    finite_diff_eps: float = 1e-4
    hessian_damping: float = 1e-2
    initial_slack: float = 1.0
    initial_dual: float = 1.0
    print_iterations: bool = False
    print_every: int = 1


@dataclass
class ExperimentConfig:
    output_dir: str = "python_outputs"
    seed: int = 0
    num_samples: int = 30
