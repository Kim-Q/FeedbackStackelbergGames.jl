from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np
from PIL import Image, ImageSequence

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

from feedback_stackelberg.scenario_highway import HighwayParameters, HighwayScenario
from feedback_stackelberg.scenario_lqr import LQRParameters


def visualize_highway(
    scenario: HighwayScenario,
    states: np.ndarray,
    output_dir: str | Path,
    base_name: str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{base_name}_python.png"
    gif_path = output_dir / f"{base_name}_python.gif"
    plot_highway_png(states, scenario.params, png_path)
    plot_highway_gif(states, scenario.params, gif_path)
    comparison_png = output_dir / f"{base_name}_comparison.png"
    comparison_gif = output_dir / f"{base_name}_comparison.gif"
    baseline_png = _repo_root() / "src" / "highway_st.png"
    baseline_gif = _repo_root() / "src" / "highway_st.gif"
    if baseline_png.exists():
        create_image_comparison(baseline_png, png_path, comparison_png)
    if baseline_gif.exists() and gif_path.exists():
        create_gif_comparison(baseline_gif, gif_path, comparison_gif)
    return {
        "png": png_path,
        "gif": gif_path,
        "comparison_png": comparison_png,
        "comparison_gif": comparison_gif,
    }


def visualize_lqr(
    params: LQRParameters,
    states: np.ndarray,
    controls: np.ndarray,
    output_dir: str | Path,
    base_name: str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{base_name}_lqr.png"
    gif_path = output_dir / f"{base_name}_lqr.gif"
    plot_lqr_png(params, states, controls, png_path)
    plot_lqr_gif(params, states, controls, gif_path)
    return {"png": png_path, "gif": gif_path}


def plot_highway_png(states: np.ndarray, params: HighwayParameters, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3, 6))
    _plot_highway_frame(ax, states, params, states.shape[0] - 1, show_legend=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_highway_gif(states: np.ndarray, params: HighwayParameters, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3, 6))
    frame_indices = list(_frame_indices(states.shape[0]))

    def update(frame_index: int) -> None:
        _plot_highway_frame(ax, states, params, frame_index, show_legend=False)

    anim = animation.FuncAnimation(fig, update, frames=frame_indices, repeat=False)
    anim.save(path, writer=animation.PillowWriter(fps=10))
    plt.close(fig)


def _plot_highway_frame(
    ax: plt.Axes,
    states: np.ndarray,
    params: HighwayParameters,
    frame_index: int,
    show_legend: bool,
) -> None:
    ax.clear()
    x1 = states[: frame_index + 1, 0]
    y1 = states[: frame_index + 1, 1]
    x2 = states[: frame_index + 1, 4]
    y2 = states[: frame_index + 1, 5]
    alphas = 0.95 ** (np.arange(len(x1), 0, -1))
    ax.scatter(x1, y1, marker="s", color="red", alpha=alphas, s=18)
    ax.scatter(x2, y2, marker="s", color="blue", alpha=alphas, s=18)
    ax.scatter([x1[-1]], [y1[-1]], marker="s", color="red", s=24, label="player 1")
    ax.scatter([x2[-1]], [y2[-1]], marker="s", color="blue", s=24, label="player 2")
    _plot_highway_road(ax, params)
    ax.set_xlim(0.0, 1.5)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlabel("p_x")
    ax.set_ylabel("p_y")
    ax.grid(False)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8)


def _plot_highway_road(ax: plt.Axes, params: HighwayParameters) -> None:
    segment_length = float(
        np.hypot(
            params.road_length - params.y_right_corner,
            params.x_right_corner - params.base_x,
        )
    )
    segment_angle = float(
        np.arctan(
            (params.x_right_corner - params.base_x)
            / (params.road_length - params.y_right_corner)
        )
    )
    radius = 0.25 * segment_length / np.sin(segment_angle)
    upper_angle = np.linspace(0.0, 2.0 * segment_angle, 40)
    lower_angle = np.linspace(2.0 * segment_angle, 0.0, 40)
    upper_curve_x = params.base_x + radius * (1.0 - np.cos(upper_angle))
    upper_curve_y = params.road_length - radius * np.sin(upper_angle)
    lower_curve_x = params.x_right_corner - radius * (1.0 - np.cos(lower_angle))
    lower_curve_y = params.y_right_corner + radius * np.sin(lower_angle)
    ax.plot([params.base_x, params.base_x], [params.road_length, 5.0], color="black", lw=2)
    ax.plot(
        [params.x_right_corner, params.x_right_corner],
        [params.y_right_corner, 0.0],
        color="black",
        lw=2,
    )
    ax.plot([0.25, 0.25], [0.0, 5.0], color="black", lw=2)
    ax.plot(upper_curve_x, upper_curve_y, color="black", lw=2)
    ax.plot(lower_curve_x, lower_curve_y, color="black", lw=2)


def plot_lqr_png(
    params: LQRParameters, states: np.ndarray, controls: np.ndarray, path: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=False)
    _plot_lqr_timeseries(axes, params, states, controls)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lqr_gif(
    params: LQRParameters, states: np.ndarray, controls: np.ndarray, path: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=False)
    state_time = np.arange(states.shape[0]) * params.dt
    control_time = np.arange(controls.shape[0]) * params.dt
    state_limits = _series_limits(states)
    control_limits = _series_limits(controls)
    frame_indices = list(_frame_indices(states.shape[0]))

    def update(frame_index: int) -> None:
        axes[0].clear()
        axes[1].clear()
        _plot_lqr_partial(
            axes,
            params,
            states,
            controls,
            frame_index,
            state_time,
            control_time,
            state_limits,
            control_limits,
        )

    anim = animation.FuncAnimation(fig, update, frames=frame_indices, repeat=False)
    anim.save(path, writer=animation.PillowWriter(fps=10))
    plt.close(fig)


def _plot_lqr_timeseries(
    axes: Sequence[plt.Axes],
    params: LQRParameters,
    states: np.ndarray,
    controls: np.ndarray,
) -> None:
    state_time = np.arange(states.shape[0]) * params.dt
    control_time = np.arange(controls.shape[0]) * params.dt
    colors_state = plt.cm.tab10(np.linspace(0.0, 1.0, states.shape[1]))
    colors_control = plt.cm.tab10(np.linspace(0.0, 1.0, controls.shape[1]))
    for idx in range(states.shape[1]):
        axes[0].plot(state_time, states[:, idx], color=colors_state[idx], label=f"x{idx + 1}")
    for idx in range(controls.shape[1]):
        axes[1].plot(control_time, controls[:, idx], color=colors_control[idx], label=f"u{idx + 1}")
    axes[0].set_ylabel("state")
    axes[1].set_ylabel("control")
    axes[1].set_xlabel("time")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.2)
    axes[1].grid(True, alpha=0.2)


def _plot_lqr_partial(
    axes: Sequence[plt.Axes],
    params: LQRParameters,
    states: np.ndarray,
    controls: np.ndarray,
    frame_index: int,
    state_time: np.ndarray,
    control_time: np.ndarray,
    state_limits: tuple[float, float],
    control_limits: tuple[float, float],
) -> None:
    colors_state = plt.cm.tab10(np.linspace(0.0, 1.0, states.shape[1]))
    colors_control = plt.cm.tab10(np.linspace(0.0, 1.0, controls.shape[1]))
    last_control_index = min(frame_index, controls.shape[0])
    for idx in range(states.shape[1]):
        axes[0].plot(
            state_time[: frame_index + 1],
            states[: frame_index + 1, idx],
            color=colors_state[idx],
        )
    for idx in range(controls.shape[1]):
        axes[1].plot(
            control_time[:last_control_index],
            controls[:last_control_index, idx],
            color=colors_control[idx],
        )
    axes[0].set_xlim(state_time[0], state_time[-1])
    axes[1].set_xlim(control_time[0], control_time[-1])
    axes[0].set_ylim(*state_limits)
    axes[1].set_ylim(*control_limits)
    axes[0].set_ylabel("state")
    axes[1].set_ylabel("control")
    axes[1].set_xlabel("time")
    axes[0].grid(True, alpha=0.2)
    axes[1].grid(True, alpha=0.2)


def create_image_comparison(left_path: Path, right_path: Path, output_path: Path) -> None:
    left_img = Image.open(left_path)
    right_img = Image.open(right_path)
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    axes[0].imshow(left_img)
    axes[0].set_title("original")
    axes[0].axis("off")
    axes[1].imshow(right_img)
    axes[1].set_title("python")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def create_gif_comparison(left_path: Path, right_path: Path, output_path: Path) -> None:
    left_frames = _load_gif_frames(left_path)
    right_frames = _load_gif_frames(right_path)
    frame_count = min(len(left_frames), len(right_frames))
    if frame_count == 0:
        return
    combined_frames = []
    for idx in range(frame_count):
        left = left_frames[idx]
        right = right_frames[idx]
        max_height = max(left.height, right.height)
        left = left.resize((int(left.width * max_height / left.height), max_height))
        right = right.resize((int(right.width * max_height / right.height), max_height))
        combined = Image.new("RGBA", (left.width + right.width, max_height), (255, 255, 255, 255))
        combined.paste(left, (0, 0))
        combined.paste(right, (left.width, 0))
        combined_frames.append(combined)
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=100,
        loop=0,
    )


def _load_gif_frames(path: Path) -> list[Image.Image]:
    with Image.open(path) as img:
        return [frame.convert("RGBA") for frame in ImageSequence.Iterator(img)]


def _frame_indices(total_frames: int, max_frames: int = 60) -> Iterable[int]:
    stride = max(1, int(np.ceil(total_frames / max_frames)))
    indices = list(range(0, total_frames, stride))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices


def _series_limits(series: np.ndarray, padding: float = 0.1) -> tuple[float, float]:
    if series.size == 0:
        return (-1.0, 1.0)
    min_val = float(np.min(series))
    max_val = float(np.max(series))
    if np.isclose(min_val, max_val):
        delta = 1.0 if min_val == 0.0 else abs(min_val) * 0.1
        return (min_val - delta, max_val + delta)
    pad = (max_val - min_val) * padding
    return (min_val - pad, max_val + pad)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
