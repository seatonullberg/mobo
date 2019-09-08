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
    def __init__(self,
                 parameters: List[Parameter],
                 qois: List[QoI],
                 manifold_dimensions: int = 0) -> None:
        self._parameter_names = [p.name for p in parameters]
        self._scaled_parameter_names = [
            "{}.scaled".format(p.name) for p in parameters
        ]
        self._qoi_names = [q.name for q in qois]
        self._error_names = ["{}.error".format(q.name) for q in qois]
        self._scaled_error_names = [
            "{}.error.scaled".format(q.name) for q in qois
        ]
        self._manifold_names = [
            "manifold_{}".format(i) for i in range(manifold_dimensions)
        ]
        self._manifold_dimensions = manifold_dimensions
        columns = (["id"] + self._parameter_names +
                   self._scaled_parameter_names + self._qoi_names +
                   self._error_names + self._scaled_error_names +
                   self._manifold_names + ["cluster_id"])
        self._df = pd.DataFrame(columns=columns)

    @classmethod
    def from_file(cls, path: str):
        # TODO
        pass

    def to_file(self, path: str) -> None:
        # TODO
        pass

    def append(self,
               iteration: int,
               parameter_values: np.ndarray,
               qoi_values: np.ndarray,
               error_values: np.ndarray,
               cluster_id: Optional[np.ndarray] = None,
               manifold_values: Optional[np.ndarray] = None,
               scaled_parameter_values: Optional[np.ndarray] = None,
               scaled_error_values: Optional[np.ndarray] = None) -> None:
        id_fmt = "{}_{}"
        id_strs = np.array([
            id_fmt.format(iteration, _id)
            for _id in range(self._df.shape[0], self._df.shape[0] +
                             len(parameter_values))
        ])
        if cluster_id is None:
            cluster_id = np.array([np.nan for _ in parameter_values])
        if scaled_parameter_values is None:
            scaled_parameter_values = np.array(
                [np.nan for _ in parameter_values])
        if scaled_error_values is None:
            scaled_error_values = np.array([np.nan for _ in error_values])
        if manifold_values is None:
            manifold_values = np.array(
                [np.nan for _ in range(self._manifold_dimensions)])
        data = {}
        data["id"] = id_strs
        data["cluster_id"] = cluster_id
        for i, p_name in enumerate(self._parameter_names):
            data[p_name] = parameter_values.T[i]
        for i, q_name in enumerate(self._qoi_names):
            data[q_name] = qoi_values.T[i]
        for i, e_name in enumerate(self._error_names):
            data[e_name] = error_values.T[i]
        for i, scaled_p_name in enumerate(self._scaled_parameter_names):
            data[scaled_p_name] = scaled_parameter_values.T[i]
        for i, scaled_e_name in enumerate(self._scaled_error_names):
            data[scaled_e_name] = scaled_error_values.T[i]
        for i, m_name in enumerate(self._manifold_names):
            data[m_name] = manifold_values.T[i]
        self._df = self._df.append(
            pd.DataFrame(data=data, columns=self._df.columns))

    def drop(self, indices: np.ndarray) -> None:
        self._df = self._df.drop(index=indices).reset_index()

    @property
    def parameter_values(self) -> np.ndarray:
        return self._df[self._parameter_names].to_numpy()

    @property
    def scaled_parameter_values(self) -> np.ndarray:
        return self._df[self._scaled_parameter_names].to_numpy()

    @property
    def qoi_values(self) -> np.ndarray:
        return self._df[self._qoi_names].to_numpy()

    @property
    def error_values(self) -> np.ndarray:
        return self._df[self._error_names].to_numpy()

    @property
    def scaled_error_values(self) -> np.ndarray:
        return self._df[self._scaled_error_names].to_numpy()

    @property
    def manifold_values(self) -> np.ndarray:
        return self._df[self._manifold_names].to_numpy()

    @property
    def ids(self) -> np.ndarray:
        return self._df["id"].to_numpy()

    @property
    def cluster_ids(self) -> np.ndarray:
        return self._df["cluster_id"].to_numpy()
