from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class HighwayParameters:
    horizon: int = 20
    dt: float = 0.05
    x_right_corner: float = 1.1
    y_right_corner: float = 2.0
    base_x: float = 0.7
    road_length: float = 4.0
    max_accel: float = 1.0
    min_accel: float = -1.0
    max_omega: float = 2.0
    min_omega: float = -2.0
    collision_radius: float = 0.4


class HighwayScenario:
    def __init__(self, params: HighwayParameters | None = None, swap_roles: bool = False):
        self.params = params or HighwayParameters()
        self.swap_roles = swap_roles
        self.nx = 8
        self.nu = 4
        self.n_players = 2
        self.players_u_index_list = ((0, 1), (2, 3))
        self.x0 = np.array(
            [0.9, 1.2, 3.5, 0.0, 0.5, 0.6, 3.8, 0.0],
            dtype=float,
        )
        self._segment_length, self._segment_angle, self._radius = self._compute_road_geometry()

    def _compute_road_geometry(self) -> Tuple[float, float, float]:
        segment_length = float(
            np.hypot(
                self.params.road_length - self.params.y_right_corner,
                self.params.x_right_corner - self.params.base_x,
            )
        )
        segment_angle = float(
            np.arctan(
                (self.params.x_right_corner - self.params.base_x)
                / (self.params.road_length - self.params.y_right_corner)
            )
        )
        radius = 0.25 * segment_length / np.sin(segment_angle)
        return segment_length, segment_angle, radius

    def initial_state(self) -> np.ndarray:
        return self.x0.copy()

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        dt = self.params.dt
        next_state = np.zeros_like(state)
        next_state[0] = state[0] + dt * state[2] * np.sin(state[3])
        next_state[1] = state[1] + dt * state[2] * np.cos(state[3])
        next_state[2] = state[2] + dt * control[0]
        next_state[3] = state[3] + dt * control[1]
        next_state[4] = state[4] + dt * state[6] * np.sin(state[7])
        next_state[5] = state[5] + dt * state[6] * np.cos(state[7])
        next_state[6] = state[6] + dt * control[2]
        next_state[7] = state[7] + dt * control[3]
        return next_state

    def stage_cost(self, state: np.ndarray, control: np.ndarray) -> float:
        leader_cost = (
            10.0 * (state[0] - 0.4) ** 2
            + 6.0 * (state[2] - state[6]) ** 2
            + 2.0 * control[0] ** 2
            + 2.0 * control[1] ** 2
        )
        follower_cost = (
            1.0 * state[7] ** 4
            + 2.0 * control[2] ** 2
            + 2.0 * control[3] ** 2
        )
        if self.swap_roles:
            leader_cost, follower_cost = follower_cost, leader_cost
        return float(leader_cost + follower_cost)

    def terminal_cost(self, state: np.ndarray) -> float:
        leader_cost = 10.0 * (state[0] - 0.4) ** 2 + 6.0 * (state[2] - state[6]) ** 2
        follower_cost = state[7] ** 4
        if self.swap_roles:
            leader_cost, follower_cost = follower_cost, leader_cost
        return float(leader_cost + follower_cost)

    def road_constraints(self, x_pos: float, y_pos: float) -> float:
        base_x = self.params.base_x
        road_length = self.params.road_length
        x_right_corner = self.params.x_right_corner
        y_right_corner = self.params.y_right_corner
        upper_circle_x = base_x + self._radius
        upper_circle_y = road_length
        lower_circle_x = x_right_corner - self._radius
        lower_circle_y = y_right_corner

        if y_pos > road_length:
            return base_x - x_pos
        if y_pos > y_right_corner:
            angle_to_upper_center = np.arctan((upper_circle_y - y_pos) / (upper_circle_x - x_pos))
            if angle_to_upper_center < 2.0 * self._segment_angle:
                distance_to_upper_center = np.hypot(x_pos - upper_circle_x, y_pos - upper_circle_y)
                return distance_to_upper_center - self._radius
            distance_to_lower_center = np.hypot(x_pos - lower_circle_x, y_pos - lower_circle_y)
            return self._radius - distance_to_lower_center
        return x_right_corner - x_pos

    def collision_avoidance(self, state: np.ndarray) -> float:
        dx = state[0] - state[4]
        dy = state[1] - state[5]
        return 2.0 * (dx * dx + dy * dy - self.params.collision_radius**2)

    def inequality_constraints(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return np.array(
            [
                state[0] - 0.25,
                state[4] - 0.25,
                self.road_constraints(state[0], state[1]),
                self.road_constraints(state[4], state[5]),
                self.collision_avoidance(state),
                self.params.max_accel - control[0],
                control[0] - self.params.min_accel,
                self.params.max_omega - control[1],
                control[1] - self.params.min_omega,
                self.params.max_accel - control[2],
                control[2] - self.params.min_accel,
                self.params.max_omega - control[3],
                control[3] - self.params.min_omega,
            ],
            dtype=float,
        )

    def terminal_inequality_constraints(self, state: np.ndarray) -> np.ndarray:
        return np.array(
            [
                state[0] - 0.25,
                state[4] - 0.25,
                self.road_constraints(state[0], state[1]),
                self.road_constraints(state[4], state[5]),
                self.collision_avoidance(state),
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
            ],
            dtype=float,
        )

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
        constraints = []
        for t in range(self.params.horizon):
            constraints.append(self.inequality_constraints(states[t], controls[t]))
        constraints.append(self.terminal_inequality_constraints(states[-1]))
        return np.concatenate(constraints, axis=0)
