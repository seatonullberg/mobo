import math
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
            # TODO: append the data here
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
        self.data: Optional[OptimizationData] = None  # set by Pipeline
        self.logger: Optional[Logger] = None          # set by Pipeline
        self.mpi: Optional[MPI.COMM_WORLD] = None     # set by Pipeline
    
    def sample(self) -> np.ndarray:
        return self.sampler.draw(self.n_samples)

    def pre_cluster_scale(self, data: np.ndarray) -> np.ndarray:
        if self.pre_cluster_scaler is None:
            err = "self.pre_cluster_scaler is not set."
            raise ValueError(err)
        else:
            return self.pre_cluster_scaler.scale(data)

    def embed(self, data: np.ndarray) -> np.ndarray:
        if self.manifold_embedder is None:
            err = "self.manifold_embedder is not set."
            raise ValueError(err)
        else:
            return self.manifold_embedder.embed(data)            

    def cluster(self, data: np.ndarray):
        if self.clusterer is None:
            err = "self.clusterer is not set."
            raise ValueError(err)
        else:
            return self.clusterer.cluster(data)

    # do the mpi stuff outside if possible
    def evaluate(self, data: np.ndarray):
        pass

    def pre_filter_scale(self, data: np.ndarray) -> np.ndarray:
        if self.pre_filter_scaler is None:
            err = "self.pre_filter_scaler is not set."
            raise ValueError(err)
        else:
            return self.pre_filter_scaler.scale(data)

    def filter(self):
        self.filter_set.apply(self.data)

    def _log(self, msg: str) -> None:
        if self.mpi.rank == 0:
            if self.logger is None:
                print(msg)
            else:
                self.logger.log(msg)

    def _n_samples_per_rank(self) -> Tuple[int, int]:
        """Determines the appropriate number of samples to process for each 
           rank.

        Notes:
            If the number of samples is not divisible by the number of ranks,
            the remainder is handled by rank 0. The first return value is rank0 
            samples the second is that of all other ranks.
        """
        n_ranks = self.mpi.Get_size()
        remainder = self.n_samples % n_ranks
        samples = math.floor(self.n_samples / n_ranks)
        rank_0_samples = remainder + samples
        return (rank_0_samples, samples)
