from mobo.parameter import Parameter
from mobo.qoi import QoI
from mobo.log import Logger
from mobo.sample import BaseSampler
from mobo.filter import BaseFilterSet
from mpi4py import MPI
import numpy as np


class Pipeline(object):
    """Implementation of an optimization pipeline.
    
    Args:
        parameters (iterable of Parameter): Parameters to optimize.
        qois (iterable of QoI): Quantities of interest to evaluate.
        segments (iterable of PipelineSegment): Segments to connect.
        logger (optional) (Logger): Centralized file logger.
        mpi_comm (optional) (mpi4py.MPI.COMM_WORLD): MPI communication object.
    """
    
    def __init__(self, parameters, qois, segments, logger=None, mpi_comm=None):
        for p in parameters:
            assert isinstance(p, Parameter)
        for q in qois:
            assert isinstance(q, QoI)
        for s in segments:
            assert type(s) is PipelineSegment
        assert type(logger) in [Logger, NoneType]
        assert type(mpi_comm) in [MPI.COMM_WORLD, NoneType]
        self._parameters = parameters
        self._qois = qois
        self._segments = segments
        self._logger = logger
        self._mpi_comm = mpi_comm

    @property
    def parameters(self):
        return self._parameters
    
    @property
    def qois(self):
        return self._qois

    @property
    def segments(self):
        return self._segments

    @property
    def logger(self):
        return self._logger

    @property
    def mpi_comm(self):
        return self._mpi_comm

    # TODO
    def execute(self):
        """Executes the optimization process."""
        pass

    def _log(self, msg):
        if self.logger is not None:
            self.logger.log(msg)


class PipelineSegment(object):
    """Implementation of a segment of an optimization pipeline.
    
    Args:
        sampler (instance of BaseSampler): Sampler to draw samples with.
        filter_set (instance of BaseFilterSet): Filters to apply.
        logger (optional) (Logger): Centralized file logger.
        TODO: splitter / subroutines

    Attributes:
        prior (numpy.ndarray): Data from a prior segment.
    """
    
    def __init__(self, sampler, filter_set, logger=None):
        assert isinstance(sampler, BaseSampler)
        assert isinstance(filter_set, BaseFilterSet)
        assert type(logger) in [Logger, NoneType]
        self._sampler = sampler
        self._filter_set = filter_set
        self._logger = logger
        self._prior = None

    @property
    def sampler(self):
        return self._sampler

    @property
    def filter_set(self):
        return self._filter_set

    @property
    def logger(self):
        return self._logger

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        assert self._prior is None  # only set this once
        assert type(value) is np.ndarray
        self._prior = value
    
    def _log(self, msg):
        if self.logger is not None:
            self.logger.log(msg)
