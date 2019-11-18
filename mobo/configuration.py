from mobo.cluster import BaseClusterer
from mobo.error import BaseErrorCalculator
from mobo.filter import BaseFilter
from mobo.log import Logger
from mobo.parameter import Parameter
from mobo.projection import BaseProjector
from mobo.qoi import QoI
from typing import List, Optional


class LocalConfiguration(object):
    """Container for information required in each iteration.
    
    Args:
        n_samples: Number of samples to draw.
        clusterer: Clustering algorithm.
        error_calculator: Error calculation scheme.
        filters: Filters to apply.
        projector: Dimensionality reduction scheme.
    """
    def __init__(self,
                 n_samples: int,
                 clusterer: BaseClusterer,
                 error_calculator: BaseErrorCalculator,
                 filters: List[BaseFilter],
                 projector: BaseProjector) -> None:
        self.n_samples = n_samples
        self.clusterer = clusterer
        self.error_calculator = error_calculator
        self.filters = filters
        self.projector = projector


class GlobalConfiguration(object):
    """Container for information required to start an optimization.
    
    Args:
        n_samples: Number of samples to start with.
        local_configurations: Configuration for each iteration.
        parameters: Parameter objects to fit.
        qois: QoI objects to evaluate.
        initial_data_path: Path to a data file to start from.
        logger: Logging utility to monitor progress of the optimization.
    """
    def __init__(self,
                 n_samples: int,
                 local_configurations: List[LocalConfiguration],
                 parameters: List[Parameter], 
                 qois: List[QoI], 
                 initial_data_path: Optional[str] = None,
                 logger: Optional[Logger] = None) -> None:
        self.n_samples = n_samples
        self.local_configurations = local_configurations
        self.parameters = parameters
        self.qois = qois
        self.initial_data_path = initial_data_path
        self.logger = logger
