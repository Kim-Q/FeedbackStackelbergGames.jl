import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np


@dataclass
class ExperimentOutput:
    states: np.ndarray
    controls: np.ndarray
    loss_history: List[List[float]]
    residual_history: List[List[float]]
    metadata: Dict[str, Any]


class ExperimentIO:
    def __init__(self, output_dir: str, run_subdir: str = None):
        self.output_dir = Path(output_dir)
        if run_subdir is not None:
            self.output_dir = self.output_dir / run_subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, output: ExperimentOutput, filename: str) -> Path:
        base_name = Path(filename).stem
        prefix = self.output_dir / base_name
        loss_matrix = self._pad_history(output.loss_history)
        residual_matrix = self._pad_history(output.residual_history)
        self._write_csv(prefix.with_name(f"{base_name}_states.csv"), output.states)
        self._write_csv(prefix.with_name(f"{base_name}_controls.csv"), output.controls)
        self._write_csv(prefix.with_name(f"{base_name}_loss_history.csv"), loss_matrix)
        self._write_csv(prefix.with_name(f"{base_name}_residual_history.csv"), residual_matrix)
        metadata_path = prefix.with_name(f"{base_name}_metadata.json")
        metadata_path.write_text(json.dumps(output.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return prefix

    def load(self, path: Union[str, Path]) -> ExperimentOutput:
        prefix = self._resolve_prefix(Path(path))
        states = self._read_csv(prefix.with_name(f"{prefix.name}_states.csv"))
        controls = self._read_csv(prefix.with_name(f"{prefix.name}_controls.csv"))
        loss_history = self._read_csv(prefix.with_name(f"{prefix.name}_loss_history.csv")).tolist()
        residual_history = self._read_csv(prefix.with_name(f"{prefix.name}_residual_history.csv")).tolist()
        metadata_path = prefix.with_name(f"{prefix.name}_metadata.json")
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return ExperimentOutput(
            states=states,
            controls=controls,
            loss_history=loss_history,
            residual_history=residual_history,
            metadata=metadata,
        )

    def _pad_history(self, history: List[List[float]]) -> np.ndarray:
        max_len = max((len(inner) for inner in history), default=0)
        padded = np.full((len(history), max_len), np.nan, dtype=float)
        for idx, inner in enumerate(history):
            if inner:
                padded[idx, : len(inner)] = np.array(inner, dtype=float)
        return padded

    def _write_csv(self, path: Path, array: np.ndarray) -> None:
        array = np.asarray(array, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        np.savetxt(path, array, delimiter=",")

    def _read_csv(self, path: Path) -> np.ndarray:
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    def _resolve_prefix(self, path: Path) -> Path:
        name = path.stem
        for suffix in ("_states", "_controls", "_loss_history", "_residual_history", "_metadata"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return path.with_name(name)


def save_multi_run_csv(
    output_dir: Union[str, Path],
    base_name: str,
    x0_list: np.ndarray,
    loss_list: np.ndarray,
    residual_list: np.ndarray,
    metadata: Dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / base_name
    x0_path = prefix.with_name(f"{base_name}_x0.csv")
    loss_path = prefix.with_name(f"{base_name}_loss.csv")
    residual_path = prefix.with_name(f"{base_name}_residual.csv")
    np.savetxt(x0_path, x0_list.reshape(x0_list.shape[0], -1), delimiter=",")
    np.savetxt(loss_path, loss_list.reshape(loss_list.shape[0], -1), delimiter=",")
    np.savetxt(residual_path, residual_list.reshape(residual_list.shape[0], -1), delimiter=",")
    metadata = {
        **metadata,
        "x0_shape": list(x0_list.shape),
        "loss_shape": list(loss_list.shape),
        "residual_shape": list(residual_list.shape),
    }
    metadata_path = prefix.with_name(f"{base_name}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return prefix


def load_multi_run_csv(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    path = Path(path)
    name = path.stem
    for suffix in ("_x0", "_loss", "_residual", "_metadata"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    prefix = path.with_name(name)
    metadata_path = prefix.with_name(f"{name}_metadata.json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    x0_shape = tuple(metadata.get("x0_shape", []))
    loss_shape = tuple(metadata.get("loss_shape", []))
    residual_shape = tuple(metadata.get("residual_shape", []))
    x0_list = np.loadtxt(prefix.with_name(f"{name}_x0.csv"), delimiter=",")
    loss_flat = np.loadtxt(prefix.with_name(f"{name}_loss.csv"), delimiter=",")
    residual_flat = np.loadtxt(prefix.with_name(f"{name}_residual.csv"), delimiter=",")
    if x0_shape:
        x0_list = x0_list.reshape(x0_shape)
    if loss_shape:
        loss_flat = loss_flat.reshape(loss_shape)
    if residual_shape:
        residual_flat = residual_flat.reshape(residual_shape)
    return x0_list, loss_flat, residual_flat, metadata
