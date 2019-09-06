# TODO: This is all garbage




from mobo.parameter import Parameter
from mobo.qoi import QoI
from mobo.log import Logger
from mobo.sample import BaseSampler
from mobo.filter import BaseFilterSet
from mobo.error import BaseErrorCalculator
from mobo.data import OptimizationData
from mpi4py import MPI
import numpy as np
from datetime import datetime


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
        assert type(logger) in [Logger, type(None)]
        assert type(mpi_comm) in [MPI.COMM_WORLD, type(None)]
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
        self._log("Beginning the optimization process...")
        for i, s in enumerate(self.segments):
            start_time = datetime.now()
            self._log("Starting segment number {}...".format(i))
            s.qois = self.qois
            s.parameters = self.parameters
            res = s.execute()  # TODO
            end_time = datetime.now()
            seconds = (end_time - start_time).total_seconds()
            msg = ("Completed segment number {} in {} seconds."
                   .format(i, seconds))
            self._log(msg)

    def _log(self, msg):
        if self.logger is not None:
            self.logger.log(msg)


class PipelineSegment(object):
    """Implementation of a segment of an optimization pipeline.
    
    Args:
        sampler (instance of BaseSampler): Sampler to draw samples with.
        filter_set (instance of BaseFilterSet): Filters to apply.
        err_calc (instance of BaseErrorCalculator): The error calculator to use.
        n_samples (int): Number of samples to draw.
        logger (optional) (Logger): Centralized file logger.

    Attributes:
        parameters (iterable of Parameter): Parameters to optimize.
        qois (iterable of QoI): Quantities of interest to evaluate.
        prior (numpy.ndarray): Data from a prior segment.
    """
    
    def __init__(self, sampler, filter_set, err_calc, n_samples, logger=None):
        assert isinstance(sampler, BaseSampler)
        assert isinstance(filter_set, BaseFilterSet)
        assert isinstance(err_calc, BaseErrorCalculator)
        assert type(n_samples) is int
        assert type(logger) in [Logger, type(None)]
        self._sampler = sampler
        self._filter_set = filter_set
        self._err_calc = err_calc
        self._n_samples = n_samples
        self._logger = logger

        # set by the pipeline
        self._parameters = None
        self._qois = None
        self._prior = None

    @property
    def sampler(self):
        return self._sampler

    @property
    def filter_set(self):
        return self._filter_set

    @property
    def err_calc(self):
        return self._err_calc

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def logger(self):
        return self._logger

    # settable attributes

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        for v in value:
            assert type(v) is Parameter
        self._parameters = value

    @property
    def qois(self):
        return self._qois

    @qois.setter
    def qois(self, value):
        for v in value:
            assert type(v) is QoI
        self._qois = value

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        assert type(value) is np.ndarray
        self._prior = value

    # TODO
    def execute(self):
        """Execute this segment of the optimization process."""
        
        if self.prior is None:
            self._log("Drawing samples without a prior distribution...")
        else:
            self._log("Drawing samples from a prior distribution...")
            self.sampler.from_prior(self.prior)

        samples = self.sampler.draw(self.n_samples)

        self._log("Evaluating the parameterizations...")
        
        if self.qois is None:
            err = "`self.qois` must be set before parameter evaluation."
            raise ValueError(err)
        
        for q in self.qois:
            msg = "Evaluating {} for all parameterizations...".format(q.name)
            self._log(msg)
            for s in samples:
                prediction = q.evaluator(s)
                print(q.name, prediction)

        # draw samples
        # (possibly from prior)
        # calculate error
        # (normalize)
        # filter results
    
    def _log(self, msg):
        if self.logger is not None:
            self.logger.log(msg)
