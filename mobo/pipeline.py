from mobo.cluster import BaseClusterer
from mobo.data import OptimizationData
from mobo.error import BaseErrorCalculator
from mobo.filter import BaseFilterSet
from mobo.log import Logger
from mobo.manifold import BaseManifoldEmbedder
from mobo.parameter import Parameter
from mobo.qoi import QoI
from mobo.sample import BaseSampler
from mobo.scale import BaseScaler
from mpi4py import MPI
import numpy as np
import os
from typing import List, Optional


class Pipeline(object):
    """An optimization pipeline.
    
    Args:
        parameters: Parameters to optimize.
        qois: Quantities of Interest to calculate.
        segments: Pipeline segments involved in the process.
        data_path_in: Path to a data file to start with.
        data_path_out: Path to a directory to write results in.
        logger: Central log file.

    Attributes:
        parameters: Parameters to optimize.
        qois: Quantities of Interest to calculate.
        segments: Pipeline segments involved in the process.
        data_path_in: Path to a data file to start with.
        data_path_out: Path to a directory to write results in.
        logger: Central log file.
        mpi: MPI COMM_WORLD object.
    """
    def __init__(self,
                 parameters: List[Parameter],
                 qois: List[QoI],
                 segments: List[PipelineSegment],
                 data_path_in: Optional[str] = None,
                 data_path_out: Optional[str] = None,
                 logger: Optional[Logger] = None) -> None:
        self.parameters = parameters
        self.qois = qois
        self.segments = segments
        self.data_path_in = data_path_in
        self.data_path_out = data_path_out
        self.logger = logger
        self.mpi = MPI.COMM_WORLD

    def execute(self) -> None:
        self._log("Executing pipeline...")
        n_segments = len(self.segments)
        if self.data_path_in is None:
            data = None
        else:
            data = OptimizationData.from_file(self.data_path_in)
        for i, segment in enumerate(self.segments):
            segment.data = data
            segment.logger = self.logger
            segment.mpi = self.mpi
            self._log("Executing segment {} of {}...".format(
                i + 1, n_segments))
            self._log("Drawing samples...")
            segment.sample()
            if segment.clusterer is not None:
                if segment.pre_cluster_scaler is not None:
                    self._log("Scaling parameters...")
                    segment.pre_cluster_scale()
                if segment.manifold_embedder is not None:
                    self._log("Manifold embedding parameters...")
                    segment.embed()
                self._log("Clustering parameters...")
                segment.cluster()
            self._log("Evaluating parameters...")
            segment.evaluate()
            if segment.pre_filter_scaler is not None:
                self._log("Scaling errors...")
                segment.pre_filter_scale()
            self._log("Filtering results...")
            segment.filter()
            data = segment.data
            if self.data_path_out is not None:
                self._log("Writing results to file...")
                path = os.path.join(self.data_path_out,
                                    "results_{}.mobo".format(i))
                data.to_file(path)

    def _log(self, msg: str) -> None:
        if self.mpi.rank == 0:
            if self.logger is None:
                print(msg)
            else:
                self.logger.log(msg)


class PipelineSegment(object):
    """A segment of the optimization pipeline.
    
    Args:
        error_calculator
        filter_set
        n_samples
        sampler
        clusterer
        manifold_embedder
        pre_cluster_scaler
        pre_filter_scaler

    Attributes:
        data
        logger
        mpi
    """
    def __init__(self,
                 error_calculator: BaseErrorCalculator,
                 filter_set: BaseFilterSet,
                 n_samples: int,
                 sampler: BaseSampler,
                 clusterer: Optional[BaseClusterer] = None,
                 manifold_embedder: Optional[BaseManifoldEmbedder] = None,
                 pre_cluster_scaler: Optional[BaseScaler] = None,
                 pre_filter_scaler: Optional[BaseScaler] = None) -> None:
        self.error_calculator = error_calculator
        self.filter_set = filter_set
        self.n_samples = n_samples
        self.sampler = sampler
        self.clusterer = clusterer
        self.manifold_embedder = manifold_embedder
        self.pre_cluster_scaler = pre_cluster_scaler
        self.pre_filter_scaler = pre_filter_scaler
        self.data: Optional[OptimizationData] = None
        self.logger: Optional[Logger] = None
        self.mpi: Optional[MPI.COMM_WORLD] = None

    def sample(self):
        pass

    def pre_cluster_scale(self):
        pass

    def embed(self):
        pass

    def cluster(self):
        pass

    def evaluate(self):
        pass

    def pre_filter_scale(self):
        pass

    def filter(self):
        pass

    def _scale(self):
        pass

    def _log(self, msg: str) -> None:
        if self.mpi.rank == 0:
            if self.logger is None:
                print(msg)
            else:
                self.logger.log(msg)
