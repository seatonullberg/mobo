import mobo
import numpy as np
import matplotlib.pyplot as plt


def polynomial(x, a=0.1234, b=0.2345, c=0.3456, d=0.4567):
    return a + b*x + c*x**2 + d*x**3


def evaluate_low(parameters):
    # evaluates the polynomial with given parameters
    x = -5
    predicted = polynomial(x, *parameters)
    return predicted


def evaluate_mid(parameters):
    # evaluate the polynomial with given parameters
    x = 0.0
    predicted = polynomial(x, *parameters)
    return predicted


def evaluate_high(parameters):
    # evaluate the polynomial with given parameters
    x = 5
    predicted = polynomial(x, *parameters)
    return predicted


if __name__ == "__main__":
    # x = np.linspace(-5, 5, 100)
    # y = [polynomial(_x) for _x in x]
    # plt.plot(x, y)
    # plt.show()

    param_a = mobo.Parameter(name="a", low=0, high=1)
    param_b = mobo.Parameter(name="b", low=0, high=1)
    param_c = mobo.Parameter(name="c", low=0, high=1)
    param_d = mobo.Parameter(name="d", low=0, high=1)

    parameters = (param_a, param_b, param_c, param_d)

    qoi_low = mobo.QoI(name="low", 
                       target=polynomial(-5.0), 
                       evaluator=evaluate_low)
    qoi_mid = mobo.QoI(name="mid",
                       target=polynomial(0.0),
                       evaluator=evaluate_mid)
    qoi_high = mobo.QoI(name="high",
                        target=polynomial(5.0),
                        evaluator=evaluate_high)
    
    qois = (qoi_low, qoi_mid, qoi_high)

    # TODO
    # - pipeline initialization
    #   - parameters
    #   - qois
    #   - logger
    #   - mpi settings 
    # - segment addition
    #   - filter set
    #   - sampler
    #   - error calculator

