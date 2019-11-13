from mobo.cluster import KmeansClusterer
from mobo.configuration import LocalConfiguration, GlobalConfiguration
from mobo.error import SquaredErrorCalculator
from mobo.filter import ParetoFilter, PercentileFilter
from mobo.optimize import Optimizer
from mobo.parameter import Parameter
from mobo.projection import PCAProjector
from mobo.qoi import QoI


# target coefficient values: 0.3, -1.2, 15.7, -18.2, 2.7
polynomial = lambda a, b, c, d, e, x: a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x 

# target: -0.993
def evaluate_pt0(params):
    x = -0.163
    return polynomial(
        params["a"], params["b"], params["c"], params["d"], params["e"], x
    )

# target: 0.0724
def evaluate_pt1(params):
    x = 0.134
    return polynomial(
        params["a"], params["b"], params["c"], params["d"], params["e"], x
    )

# target: 0.4944
def evaluate_pt2(params):
    x = 1.095
    return polynomial(
        params["a"], params["b"], params["c"], params["d"], params["e"], x
    )


if __name__ == "__main__":
    # construct parameters
    parameters = [
        Parameter("a", -1.0, 1.0), Parameter("b", -2.0, 0.0),
        Parameter("c", 12.0, 16.0), Parameter("d", -20.0, -16.0),
        Parameter("e", 1.0, 3.0)
    ]
    # construct qois
    qois = [
        QoI("pt0", evaluate_pt0, -0.993),
        QoI("pt1", evaluate_pt1, 0.0724),
        QoI("pt2", evaluate_pt2, 0.4944)
    ]
    # construct local configurations
    n_iterations = 5
    n_samples = 20000
    clusterer = KmeansClusterer(n_clusters=2)
    error_calculator = SquaredErrorCalculator()
    filters = [ParetoFilter(), PercentileFilter(99)]
    projector = PCAProjector()
    local_config = LocalConfiguration(n_samples,
                                      clusterer,
                                      error_calculator,
                                      filters,
                                      projector)
    local_configurations = [local_config for _ in range(n_iterations)]
    # construct global configuration
    global_config = GlobalConfiguration(n_samples, 
                                        n_iterations, 
                                        local_configurations, 
                                        parameters, 
                                        qois)
    # construct optimizer
    optimizer = Optimizer(global_config)
    optimizer() # begin optimization
    # TODO: visualize results
