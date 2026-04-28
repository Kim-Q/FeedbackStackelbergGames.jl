from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from feedback_stackelberg.config import ExperimentConfig, PDIPConfig
from feedback_stackelberg.io_utils import ExperimentIO, ExperimentOutput
from feedback_stackelberg.pdip_solver import PDIPSolver
from feedback_stackelberg.scenario_highway import HighwayParameters, HighwayScenario
from feedback_stackelberg.scenario_lqr import LQRParameters, LQRScenario, load_lqr_overrides


@dataclass
class MultiRunOutput:
    x0_list: np.ndarray
    loss_list: np.ndarray
    residual_list: np.ndarray
    metadata: Dict[str, Any]


class BaseExperiment:
    def __init__(self, config: ExperimentConfig, solver_config: PDIPConfig):
        self.config = config
        self.solver_config = solver_config
        self.io = ExperimentIO(config.output_dir)

    def run(self) -> Any:
        raise NotImplementedError


class HighwayExperiment(BaseExperiment):
    def __init__(
        self,
        config: ExperimentConfig,
        solver_config: PDIPConfig,
        save_output: bool = True,
        swap_roles: bool = False,
    ):
        super().__init__(config, solver_config)
        self.save_output = save_output
        self.swap_roles = swap_roles

    def run(self) -> ExperimentOutput:
        scenario = HighwayScenario(params=HighwayParameters(), swap_roles=self.swap_roles)
        solver = PDIPSolver(self.solver_config)
        result = solver.solve(scenario, scenario.initial_state())
        metadata = {
            "environment": "highway",
            "swap_roles": self.swap_roles,
            "horizon": scenario.params.horizon,
        }
        output = ExperimentOutput(
            states=result.states,
            controls=result.controls,
            loss_history=result.loss_history,
            residual_history=result.residual_history,
            metadata=metadata,
        )
        if self.save_output:
            filename = self._build_filename("highway")
            self.io.save(output, filename)
        return output

    def _build_filename(self, prefix: str) -> str:
        timestamp = np.datetime_as_string(np.datetime64("now"), unit="s").replace(":", "-")
        return f"{prefix}_{timestamp}.npz"


class LQRExperiment(BaseExperiment):
    def __init__(
        self,
        config: ExperimentConfig,
        solver_config: PDIPConfig,
        lqr_params_path: str | None = None,
        lqr_dynamics: str = "linear",
        lqr_horizon: int | None = None,
        lqr_dt: float | None = None,
    ):
        super().__init__(config, solver_config)
        self.lqr_params_path = lqr_params_path
        self.lqr_dynamics = lqr_dynamics
        self.lqr_horizon = lqr_horizon
        self.lqr_dt = lqr_dt

    def run(self) -> ExperimentOutput:
        params = LQRParameters()
        if self.lqr_params_path:
            params.apply_overrides(load_lqr_overrides(self.lqr_params_path))
        if self.lqr_horizon is not None:
            params.horizon = self.lqr_horizon
        if self.lqr_dt is not None:
            params.dt = self.lqr_dt
        if self.lqr_dynamics:
            params.dynamics = self.lqr_dynamics
        params.normalize()
        scenario = LQRScenario(params=params)
        solver = PDIPSolver(self.solver_config)
        result = solver.solve(scenario, scenario.initial_state())
        metadata = {
            "environment": "lqr",
            "dynamics": params.dynamics,
            "horizon": params.horizon,
            "state_dim": scenario.nx,
            "control_dim": scenario.nu,
        }
        output = ExperimentOutput(
            states=result.states,
            controls=result.controls,
            loss_history=result.loss_history,
            residual_history=result.residual_history,
            metadata=metadata,
        )
        filename = self._build_filename("lqr")
        self.io.save(output, filename)
        return output


class HighwayWithoutSavingExperiment(HighwayExperiment):
    def __init__(self, config: ExperimentConfig, solver_config: PDIPConfig):
        super().__init__(config, solver_config, save_output=False)


class HighwaySwitchLeaderExperiment(HighwayExperiment):
    def __init__(self, config: ExperimentConfig, solver_config: PDIPConfig):
        super().__init__(config, solver_config, save_output=True, swap_roles=True)


class HighwayMultiRunExperiment(BaseExperiment):
    def run(self) -> MultiRunOutput:
        rng = np.random.default_rng(self.config.seed)
        scenario = HighwayScenario(params=HighwayParameters())
        solver = PDIPSolver(self.solver_config)
        x0_list = []
        loss_list = []
        residual_list = []
        for _ in range(self.config.num_samples):
            noise = 0.2 * (rng.random(scenario.nx) - 0.5)
            x0 = scenario.initial_state() + noise
            x0_list.append(x0)
            result = solver.solve(scenario, x0)
            loss_list.append(self._pad_history(result.loss_history))
            residual_list.append(self._pad_history(result.residual_history))
        loss_array = np.stack(loss_list, axis=0)
        residual_array = np.stack(residual_list, axis=0)
        output = MultiRunOutput(
            x0_list=np.stack(x0_list, axis=0),
            loss_list=loss_array,
            residual_list=residual_array,
            metadata={
                "environment": "highway_multi_run",
                "num_samples": self.config.num_samples,
                "horizon": scenario.params.horizon,
            },
        )
        self._save_multi_run(output)
        return output

    def _pad_history(self, history: List[List[float]]) -> np.ndarray:
        max_len = max((len(inner) for inner in history), default=0)
        padded = np.full((len(history), max_len), np.nan, dtype=float)
        for idx, inner in enumerate(history):
            if inner:
                padded[idx, : len(inner)] = np.array(inner, dtype=float)
        return padded

    def _save_multi_run(self, output: MultiRunOutput) -> None:
        timestamp = np.datetime_as_string(np.datetime64("now"), unit="s").replace(":", "-")
        filename = f"highway_multi_run_{timestamp}.npz"
        path = Path(self.config.output_dir) / filename
        np.savez(
            path,
            x0_list=output.x0_list,
            loss_list=output.loss_list,
            residual_list=output.residual_list,
            metadata=output.metadata,
        )


class HighwayDataProcessingExperiment(BaseExperiment):
    def __init__(self, config: ExperimentConfig, solver_config: PDIPConfig, input_path: str):
        super().__init__(config, solver_config)
        self.input_path = Path(input_path)

    def run(self) -> Dict[str, Any]:
        output = self.io.load(self.input_path)
        summary = self._build_summary(output)
        summary_path = Path(self.config.output_dir) / "highway_summary.json"
        summary_path.write_text(self._format_summary(summary), encoding="utf-8")
        self._plot_trajectory(output, Path(self.config.output_dir) / "highway_trajectory.png")
        return summary

    def _build_summary(self, output: ExperimentOutput) -> Dict[str, Any]:
        final_state = output.states[-1].tolist()
        return {
            "final_state": final_state,
            "loss_history": output.loss_history,
            "residual_history": output.residual_history,
        }

    def _format_summary(self, summary: Dict[str, Any]) -> str:
        import json

        return json.dumps(summary, ensure_ascii=False, indent=2)

    def _plot_trajectory(self, output: ExperimentOutput, path: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            return
        states = output.states
        plt.figure(figsize=(3, 6))
        plt.plot(states[:, 0], states[:, 1], color="red", label="player 1")
        plt.plot(states[:, 4], states[:, 5], color="blue", label="player 2")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class HighwayMultiRunPlotExperiment(BaseExperiment):
    def __init__(self, config: ExperimentConfig, solver_config: PDIPConfig, input_path: str):
        super().__init__(config, solver_config)
        self.input_path = Path(input_path)

    def run(self) -> Dict[str, Any]:
        data = np.load(self.input_path, allow_pickle=True)
        loss_list = data["loss_list"]
        mean_loss = np.nanmean(loss_list, axis=0)
        std_loss = np.nanstd(loss_list, axis=0)
        summary = {
            "mean_loss": mean_loss.tolist(),
            "std_loss": std_loss.tolist(),
        }
        summary_path = Path(self.config.output_dir) / "highway_multi_run_summary.json"
        summary_path.write_text(self._format_summary(summary), encoding="utf-8")
        self._plot_loss(mean_loss, std_loss, Path(self.config.output_dir) / "highway_multi_run_loss.png")
        return summary

    def _format_summary(self, summary: Dict[str, Any]) -> str:
        import json

        return json.dumps(summary, ensure_ascii=False, indent=2)

    def _plot_loss(self, mean_loss: np.ndarray, std_loss: np.ndarray, path: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            return
        iterations = np.arange(1, mean_loss.shape[1] + 1)
        plt.figure(figsize=(4, 3))
        plt.plot(iterations, mean_loss[0], color="black")
        plt.fill_between(
            iterations,
            mean_loss[0] - std_loss[0],
            mean_loss[0] + std_loss[0],
            color="gray",
            alpha=0.2,
        )
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, solver_config: PDIPConfig):
        self.config = config
        self.solver_config = solver_config

    def run(
        self,
        environment: str,
        input_path: str | None = None,
        lqr_params_path: str | None = None,
        lqr_dynamics: str = "linear",
        lqr_horizon: int | None = None,
        lqr_dt: float | None = None,
    ) -> Any:
        if environment == "highway":
            experiment = HighwayExperiment(self.config, self.solver_config)
        elif environment == "highway_without_saving_data":
            experiment = HighwayWithoutSavingExperiment(self.config, self.solver_config)
        elif environment == "highway_switch_leader_and_follower":
            experiment = HighwaySwitchLeaderExperiment(self.config, self.solver_config)
        elif environment == "highway_multi_run":
            experiment = HighwayMultiRunExperiment(self.config, self.solver_config)
        elif environment == "highway_data_processing":
            experiment = HighwayDataProcessingExperiment(
                self.config, self.solver_config, self._require_input(input_path)
            )
        elif environment == "highway_multi_run_plot":
            experiment = HighwayMultiRunPlotExperiment(
                self.config, self.solver_config, self._require_input(input_path)
            )
        elif environment == "lqr":
            experiment = LQRExperiment(
                self.config,
                self.solver_config,
                lqr_params_path=lqr_params_path,
                lqr_dynamics=lqr_dynamics,
                lqr_horizon=lqr_horizon,
                lqr_dt=lqr_dt,
            )
        else:
            raise ValueError(f"Unsupported environment: {environment}")
        return experiment.run()

    def _require_input(self, input_path: str | None) -> str:
        if not input_path:
            raise ValueError("input_path is required for this environment")
        return input_path
