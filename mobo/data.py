from mobo.parameter import Parameter
from mobo.qoi import QoI
import numpy as np
import pandas as pd
from typing import List, Optional


class OptimizationData(object):
    """Data which gets passed through a pipeline.

    Args:
        paramters: List of Parameter objects.
        qois: List of QoI objects.
    """
    def __init__(self, parameters: List[Parameter], qois: List[QoI]) -> None:
        self._parameter_names = [p.name for p in parameters]
        self._qoi_names = [q.name for q in qois]
        self._error_names = ["{}.error".format(q.name) for q in qois]
        columns = (
            ["id"] + self._parameter_names + self._qoi_names
            + self._error_names + ["cluster_id"]
        )
        self._df = pd.DataFrame(columns=columns)

    @classmethod
    def from_file(cls, path: str):
        # TODO
        pass

    def to_file(self, path: str) -> None:
        # TODO
        pass

    def append(self, iteration: int,
               parameter_values: np.ndarray,
               qoi_values: np.ndarray,
               error_values: np.ndarray,
               cluster_id: Optional[np.ndarray] = None) -> None:
        id_fmt = "{}_{}"
        id_strs = np.array([
            id_fmt.format(iteration, _id)
            for _id in range(self._df.shape[0],
                             self._df.shape[0] + len(parameter_values))
        ])
        if cluster_id is None:
            cluster_id = np.array([np.nan for _ in parameter_values])
        data = {}
        data["id"] = id_strs
        for i, p_name in enumerate(self._parameter_names):
            data[p_name] = parameter_values.T[i]
        for i, q_name in enumerate(self._qoi_names):
            data[q_name] = qoi_values.T[i]
        for i, e_name in enumerate(self._error_names):
            data[e_name] = error_values.T[i]
        data["cluster_id"] = cluster_id
        self._df = self._df.append(
            pd.DataFrame(data=data, columns=self._df.columns)
        )

    def drop(self, indices: np.ndarray) -> None:
        self._df = self._df.drop(index=indices).reset_index()

    @property
    def parameter_values(self) -> np.ndarray:
        return self._df[self._parameter_names].to_numpy()

    @property
    def qoi_values(self) -> np.ndarray:
        return self._df[self._qoi_names].to_numpy()

    @property
    def error_values(self) -> np.ndarray:
        return self._df[self._error_names].to_numpy()

    @property
    def ids(self) -> np.ndarray:
        return self._df["id"].to_numpy()

    @property
    def cluster_ids(self) -> np.ndarray:
        return self._df["cluster_id"].to_numpy()
