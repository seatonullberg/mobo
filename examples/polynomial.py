from mobo.cluster import KmeansClusterer
from mobo.configuration import LocalConfiguration, GlobalConfiguration
from mobo.error import SquaredErrorCalculator
from mobo.filter import ParetoFilter, PercentileFilter
from mobo.optimize import Optimizer
from mobo.parameter import Parameter
from mobo.projection import PCAProjector
from mobo.qoi import QoI

# target coefficients: 0.3, -1.2, 0.5
polynomial = lambda a, b, c, x: a*x**3 + b*x**2 + c*x

# target: -0.062
def evaluate_pt0(params):
    x = -0.1
    return polynomial(params["a"], params["b"], params["c"], x)

# target: 0.05
def evaluate_pt1(params):
    x = 0.3
    return polynomial(params["a"], params["b"], params["c"], x)

# target: -1.4
def evaluate_pt2(params):
    x = 2.0
    return polynomial(params["a"], params["b"], params["c"], x)

# target: 2.0
def evaluate_pt3(params):
    x = 4.0
    return polynomial(params["a"], params["b"], params["c"], x)


if __name__ == "__main__":
    ###################
    #  CONFIGURATION  #
    ###################
    
    # construct parameters
    parameters = [
        Parameter("a", -1.0, 1.0), 
        Parameter("b", -2.0, 0.0),
        Parameter("c", 0.0, 1.0)
    ]
    
    # construct qois
    qois = [
        QoI("pt0", evaluate_pt0, -0.062),
        QoI("pt1", evaluate_pt1, 0.05),
        QoI("pt2", evaluate_pt2, -1.4),
        QoI("pt3", evaluate_pt3, 2.0)
    ]
    
    # global settings
    n_iterations = 5
    n_samples = 15000
    
    # construct local configurations
    clusterer = KmeansClusterer(n_clusters=2)
    error_calculator = SquaredErrorCalculator()
    filters = [ParetoFilter(), PercentileFilter()]
    projector = PCAProjector()
    local_config = LocalConfiguration(
        n_samples, clusterer, error_calculator, filters, projector
    )
    local_configurations = [local_config for _ in range(n_iterations)]
    
    # construct global configuration
    global_config = GlobalConfiguration(
        n_samples, n_iterations, local_configurations, parameters, qois
    )
    
    ##################
    #  OPTIMIZATION  #
    ##################

    # construct optimizer
    optimizer = Optimizer(global_config)
    
    # optimize
    optimizer()

    ###################
    #  VISUALIZATION  #
    ###################

    # extra imports
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # load final data file
    final_data = pd.read_csv("mobo_iteration_4.csv")

    # construct plot
    fig, ax = plt.subplots()
    x = np.arange(-1.25, 4.25, 0.1)
    for _, p in final_data[optimizer.parameter_names].iterrows():
        y = [polynomial(p["a"], p["b"], p["c"], _x) for _x in x]
        ax.plot(x, y, alpha=0.2, color="green", zorder=0)
    exact_y = [polynomial(0.3, -1.2, 0.5, _x) for _x in x]
    ax.plot(x, exact_y, color="black", zorder=1)
    test_x = [-0.1, 0.3, 2.0, 4.0]
    test_y = [-0.062, 0.05, -1.4, 2.0]
    ax.scatter(test_x, test_y, color="black", zorder=2)
    ax.set_ylim(-4, 4)
    ax.set_title(r"$y = 0.3x^3 - 1.2x^2 + 0.5x$")
    plt.tight_layout()
    plt.savefig("polynomial.png")
