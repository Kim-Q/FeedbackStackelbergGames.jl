import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class ExperimentOutput:
    states: np.ndarray
    controls: np.ndarray
    loss_history: List[List[float]]
    residual_history: List[List[float]]
    metadata: Dict[str, Any]


class ExperimentIO:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, output: ExperimentOutput, filename: str) -> Path:
        path = self.output_dir / filename
        loss_matrix = self._pad_history(output.loss_history)
        residual_matrix = self._pad_history(output.residual_history)
        metadata_json = json.dumps(output.metadata, ensure_ascii=False)
        np.savez(
            path,
            states=output.states,
            controls=output.controls,
            loss_history=loss_matrix,
            residual_history=residual_matrix,
            metadata=metadata_json,
        )
        return path

    def load(self, path: str | Path) -> ExperimentOutput:
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data["metadata"]))
        loss_history = data["loss_history"].tolist()
        residual_history = data["residual_history"].tolist()
        return ExperimentOutput(
            states=data["states"],
            controls=data["controls"],
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
