from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np


@dataclass
class LQRParameters:
    horizon: int = 30
    dt: float = 0.05
    dynamics: str = "linear"
    A: np.ndarray = field(default_factory=lambda: np.array([[0.0, 1.0], [-1.0, -0.1]], dtype=float))
    B_leader: np.ndarray = field(default_factory=lambda: np.array([[0.0], [1.0]], dtype=float))
    B_follower: np.ndarray = field(default_factory=lambda: np.array([[0.0], [0.5]], dtype=float))
    Q_leader: np.ndarray = field(default_factory=lambda: np.diag([1.0, 0.2]).astype(float))
    Q_follower: np.ndarray = field(default_factory=lambda: np.diag([0.8, 0.5]).astype(float))
    Q_terminal_leader: np.ndarray | None = None
    Q_terminal_follower: np.ndarray | None = None
    R_leader: np.ndarray = field(default_factory=lambda: np.array([[0.5]], dtype=float))
    R_follower: np.ndarray = field(default_factory=lambda: np.array([[0.3]], dtype=float))
    R_leader_follower: np.ndarray = field(default_factory=lambda: np.array([[0.1]], dtype=float))
    R_follower_leader: np.ndarray = field(default_factory=lambda: np.array([[0.1]], dtype=float))
    Theta_leader: np.ndarray = field(default_factory=lambda: np.array([[0.05]], dtype=float))
    Theta_follower: np.ndarray = field(default_factory=lambda: np.array([[0.05]], dtype=float))
    x0: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=float))
    f_func: Callable[[np.ndarray], np.ndarray] | None = None
    g_leader_func: Callable[[np.ndarray], np.ndarray] | None = None
    g_follower_func: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.normalize()

    def normalize(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.B_leader = np.asarray(self.B_leader, dtype=float)
        self.B_follower = np.asarray(self.B_follower, dtype=float)
        self.Q_leader = np.asarray(self.Q_leader, dtype=float)
        self.Q_follower = np.asarray(self.Q_follower, dtype=float)
        self.R_leader = np.asarray(self.R_leader, dtype=float)
        self.R_follower = np.asarray(self.R_follower, dtype=float)
        self.R_leader_follower = np.asarray(self.R_leader_follower, dtype=float)
        self.R_follower_leader = np.asarray(self.R_follower_leader, dtype=float)
        self.Theta_leader = np.asarray(self.Theta_leader, dtype=float)
        self.Theta_follower = np.asarray(self.Theta_follower, dtype=float)
        self.x0 = np.asarray(self.x0, dtype=float)
        if self.Q_terminal_leader is None:
            self.Q_terminal_leader = self.Q_leader.copy()
        else:
            self.Q_terminal_leader = np.asarray(self.Q_terminal_leader, dtype=float)
        if self.Q_terminal_follower is None:
            self.Q_terminal_follower = self.Q_follower.copy()
        else:
            self.Q_terminal_follower = np.asarray(self.Q_terminal_follower, dtype=float)

    def apply_overrides(self, overrides: Dict[str, Any]) -> "LQRParameters":
        for key, value in overrides.items():
            setattr(self, key, value)
        self.normalize()
        return self


class LQRScenario:
    def __init__(self, params: LQRParameters | None = None):
        self.params = params or LQRParameters()
        self.nx = int(self.params.A.shape[0])
        self._nu_leader = int(self.params.B_leader.shape[1])
        self._nu_follower = int(self.params.B_follower.shape[1])
        self.nu = self._nu_leader + self._nu_follower
        self.n_players = 2
        self.players_u_index_list = (
            tuple(range(self._nu_leader)),
            tuple(range(self._nu_leader, self.nu)),
        )

    def initial_state(self) -> np.ndarray:
        return self.params.x0.copy()

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        u_leader, u_follower = self._split_controls(control)
        if self.params.dynamics == "linear":
            return self._linear_dynamics(state, u_leader, u_follower)
        return self._nonlinear_dynamics(state, u_leader, u_follower)

    def stage_cost(self, state: np.ndarray, control: np.ndarray) -> float:
        u_leader, u_follower = self._split_controls(control)
        leader_cost = self._player_cost(
            self.params.Q_leader,
            self.params.R_leader,
            self.params.R_leader_follower,
            self.params.Theta_leader,
            state,
            u_leader,
            u_follower,
        )
        follower_cost = self._player_cost(
            self.params.Q_follower,
            self.params.R_follower,
            self.params.R_follower_leader,
            self.params.Theta_follower,
            state,
            u_follower,
            u_leader,
        )
        return float(leader_cost + follower_cost)

    def terminal_cost(self, state: np.ndarray) -> float:
        leader_cost = 0.5 * float(state @ self.params.Q_terminal_leader @ state)
        follower_cost = 0.5 * float(state @ self.params.Q_terminal_follower @ state)
        return float(leader_cost + follower_cost)

    def inequality_constraints(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return np.empty((0,), dtype=float)

    def terminal_inequality_constraints(self, state: np.ndarray) -> np.ndarray:
        return np.empty((0,), dtype=float)

    def forward_simulation(self, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = np.zeros((self.params.horizon + 1, self.nx), dtype=float)
        states[0] = x0
        for t in range(self.params.horizon):
            states[t + 1] = self.dynamics(states[t], controls[t])
        return states

    def total_cost(self, states: np.ndarray, controls: np.ndarray) -> float:
        total = 0.0
        for t in range(self.params.horizon):
            total += self.stage_cost(states[t], controls[t])
        total += self.terminal_cost(states[-1])
        return float(total)

    def collect_constraints(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        return np.empty((0,), dtype=float)

    def _split_controls(self, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        control = np.asarray(control, dtype=float).reshape(-1)
        leader = control[: self._nu_leader]
        follower = control[self._nu_leader :]
        return leader, follower

    def _linear_dynamics(
        self, state: np.ndarray, u_leader: np.ndarray, u_follower: np.ndarray
    ) -> np.ndarray:
        x_dot = self.params.A @ state + self.params.B_leader @ u_leader + self.params.B_follower @ u_follower
        return state + self.params.dt * x_dot

    def _nonlinear_dynamics(
        self, state: np.ndarray, u_leader: np.ndarray, u_follower: np.ndarray
    ) -> np.ndarray:
        f_func = self.params.f_func or self._default_f
        g_leader = self.params.g_leader_func or self._default_g_leader
        g_follower = self.params.g_follower_func or self._default_g_follower
        x_dot = f_func(state) + g_leader(state) @ u_leader + g_follower(state) @ u_follower
        return state + self.params.dt * x_dot

    def _default_f(self, state: np.ndarray) -> np.ndarray:
        return self.params.A @ state + 0.1 * np.sin(state)

    def _default_g_leader(self, state: np.ndarray) -> np.ndarray:
        return self.params.B_leader

    def _default_g_follower(self, state: np.ndarray) -> np.ndarray:
        return self.params.B_follower

    def _player_cost(
        self,
        Q: np.ndarray,
        R_ii: np.ndarray,
        R_ij: np.ndarray,
        Theta: np.ndarray,
        state: np.ndarray,
        u_i: np.ndarray,
        u_j: np.ndarray,
    ) -> float:
        state_term = float(state @ Q @ state)
        control_term = float(u_i @ R_ii @ u_i + u_j @ R_ij @ u_j + 2.0 * u_i @ Theta @ u_j)
        return 0.5 * (state_term + control_term)


def load_lqr_overrides(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return _load_overrides_from_npz(data)
    if path.suffix == ".json":
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        return _load_overrides_from_json(raw)
    raise ValueError(f"Unsupported LQR parameter file: {path}")


def _load_overrides_from_npz(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key in data.files:
        value = data[key]
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if key == "dynamics":
            overrides[key] = str(value)
        else:
            overrides[key] = value
    return overrides


def _load_overrides_from_json(raw: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, value in raw.items():
        if key == "dynamics":
            overrides[key] = str(value)
        elif isinstance(value, list):
            overrides[key] = np.asarray(value, dtype=float)
        else:
            overrides[key] = value
    return overrides
