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
from typing import List, Optional, Tuple

# TODO: Make a decorator to control serial operations
# `with_rank_0`


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
        # read data from file if provided
        if self.data_path_in is None:
            data = None
        else:
            data = OptimizationData.from_file(self.data_path_in)
        # iterate over each segment in the pipeline
        for i, segment in enumerate(self.segments):
            # pass the data, logger, and mpi objects to the segment
            segment.data = data
            segment.logger = self.logger
            segment.mpi = self.mpi
            self._log("Executing segment {} of {}...".format(
                i + 1, n_segments))
            # draw initial parameters
            self._log("Drawing samples...")
            parameters = segment.sample()
            # initialize optional values
            scaled_parameters: Optional[np.ndarray] = None
            manifold_embedding: Optional[np.ndarray] = None
            cluster_ids: Optional[np.ndarray] = None
            scaled_errors: Optional[np.ndarray] = None
            # check optional steps
            if segment.clusterer is not None:
                # scale the parameters if desired
                if segment.pre_cluster_scaler is not None:
                    self._log("Scaling parameters...")
                    scaled_parameters = segment.pre_cluster_scale(parameters)
                # calculate manifold if desired
                # manifold can be done over various data or not at all
                # hierarchy is as follows:
                #   1) Scaled parameters
                #   2) Raw parameters
                if segment.manifold_embedder is not None:
                    self._log("Manifold embedding parameters...")
                    if scaled_parameters is None:
                        manifold_embedding = segment.embed(parameters)
                    else:
                        manifold_embedding = segment.embed(scaled_parameters)
                # clustering can be done over various datas or not at all
                # hierarchy is as follows:
                #   1) Manifold embedding
                #   2) Scaled parameter values
                #   3) Raw parameter values
                self._log("Clustering parameters...")
                if manifold_embedding is not None:
                    cluster_ids = segment.clusterer.cluster(manifold_embedding)
                elif scaled_parameters is not None:
                    cluster_ids = segment.clusterer.cluster(scaled_parameters)
                else:
                    cluster_ids = segment.clusterer.cluster(parameters)
            # Evaluate the QoIs based on raw parameter values
            self._log("Evaluating parameters...")
            qois = segment.evaluate(parameters)
            # TODO: Calculate error between results and targets
            # ??? not sure how to package targets
            self._log("Calculating errors...")
            #errors = segment.error_calculator.calculate()
            # scale the errors if desired
            if segment.pre_filter_scaler is not None:
                self._log("Scaling errors...")
                scaled_errors = segment.pre_filter_scale(errors)
            self._log("Filtering results...")
            # concat the data for filtering
            # catch possible data failure
            if data is None:
                err = "data is not set"
                raise ValueError(err)
            data.append(iteration=i,
                        parameter_values=parameters,
                        qoi_values=qois,
                        error_values=errors,
                        cluster_id=cluster_ids,
                        manifold_values=manifold_embedding,
                        scaled_parameter_values=scaled_parameters,
                        scaled_error_values=scaled_errors)
            # TODO: filter needs to know if scaled errors should be used
            segment.filter()
            # update the OptimizationData
            data = segment.data
            # Optionally (recommended) write the iteration result to file
            if self.data_path_out is not None:
                self._log("Writing results to file...")
                path = os.path.join(self.data_path_out,
                                    "results_{}.csv".format(i))
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
        self._data: Optional[OptimizationData] = None  # set by Pipeline
        self._logger: Optional[Logger] = None  # set by Pipeline
        self._mpi: Optional[MPI.COMM_WORLD] = None  # set by Pipeline

    def sample(self) -> np.ndarray:
        return self.sampler.draw(self.n_samples)

    def pre_cluster_scale(self, data: np.ndarray) -> np.ndarray:
        if self.pre_cluster_scaler is None:
            err = "self.pre_cluster_scaler is not set."
            raise ValueError(err)
        return self.pre_cluster_scaler.scale(data)

    def embed(self, data: np.ndarray) -> np.ndarray:
        if self.manifold_embedder is None:
            err = "self.manifold_embedder is not set."
            raise ValueError(err)
        return self.manifold_embedder.embed(data)

    def cluster(self, data: np.ndarray) -> np.ndarray:
        if self.clusterer is None:
            err = "self.clusterer is not set."
            raise ValueError(err)
        return self.clusterer.cluster(data)

    # lots of mpi work to do
    def evaluate(self, data: np.ndarray, qois: List[QoI]) -> np.ndarray:
        pass

    def pre_filter_scale(self, data: np.ndarray) -> np.ndarray:
        if self.pre_filter_scaler is None:
            err = "self.pre_filter_scaler is not set."
            raise ValueError(err)
        return self.pre_filter_scaler.scale(data)

    def filter(self) -> None:
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
            samples the second is that of all other ranks. Maybe not the best
            algorithm, but hey, I'm not a CS student.
        """
        n_ranks = self.mpi.Get_size()
        remainder = self.n_samples % n_ranks
        samples = math.floor(self.n_samples / n_ranks)
        rank_0_samples = remainder + samples
        return (rank_0_samples, samples)

    # Things set by Pipeline are properties to prevent None check everywhere
    # when there will actually be no chance of being None at runtime.
    # Logger doesn't count because it is actually optional.

    @property
    def data(self) -> OptimizationData:
        if self._data is None:
            err = "self.data is not set."
            raise ValueError(err)
        return self._data

    @data.setter
    def data(self, value: OptimizationData) -> None:
        self._data = value

    @property
    def mpi(self) -> MPI.COMM_WORLD:
        if self._mpi is None:
            err = "self.mpi is not set."
            raise ValueError(err)
        return self._mpi

    @mpi.setter
    def mpi(self, value: MPI.COMM_WORLD) -> None:
        self._mpi = value
