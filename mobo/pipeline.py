

class Pipeline(object):
    """Implementation of an optimization pipeline.
    
    Args:
        parameters (iterable of Parameter): Parameters to optimize.
        qois (iterable of QoI): Quantities of interest to evaluate.
        logger (Logger): Centralized file logger.
        mpi (): TODO
    """
    pass


class PipelineSegment(object):
    """Implementation of a segment of an optimization pipeline.
    
    Args:
        filter_set (instance of BaseFilterSet): Filters to apply post-iteration.
        sampler (instance of BaseSampler): Sampler to draw samples with.
        splitter (): TODO ?? subroutines
    """
    pass