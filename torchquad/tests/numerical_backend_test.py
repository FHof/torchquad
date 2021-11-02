#!/usr/bin/env python3
import sys

sys.path.append("../")

from autoray import numpy as anp
from autoray import infer_backend, get_dtype_name

from integration.trapezoid import Trapezoid
from integration.simpson import Simpson
from integration.boole import Boole
from integration.monte_carlo import MonteCarlo
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision


# Setup for tensorflow so that Newton Cotes works
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def run_simple_integrations(backend, err_max=18e-16, dtype_name="float64"):
    """
    Integrate a simple 2D constant function and a linear 2D function to check
    the following:
    * The integration rules work with the numerical backend
    * The points have the correct format
    * MonteCarlo and the Newton Cotes composite integrators integrate a
      constant function (almost) exactly.
    * The Newton Cotes composite integrators integrate a polynomial of
      degree 1 exactly.
    """
    # Newton Cotes composite integrators (nc)
    integrators_nc = [Trapezoid(), Simpson(), Boole()]
    Ns_nc = [13 ** 2, 13 ** 2, 13 ** 2]

    integrators_all = integrators_nc + [MonteCarlo()]
    Ns_all = Ns_nc + [20]

    def fn_const(x):
        assert infer_backend(x) == backend
        if dtype_name is not None:
            assert get_dtype_name(x) == dtype_name
        assert len(x.shape) == 2 and x.shape[1] == 2
        return 0.0 * x[:, 0] - 2.0

    def fn_linear(x):
        assert infer_backend(x) == backend
        if dtype_name is not None:
            assert get_dtype_name(x) == dtype_name
        assert len(x.shape) == 2 and x.shape[1] == 2
        return 3.0 * x[:, 0] + x[:, 1]

    for integr, N in zip(integrators_all, Ns_all):
        print(
            f"[2mTesting {type(integr).__name__} with {backend} and a constant"
            " function[m"
        )
        result = integr.integrate(
            fn=fn_const,
            dim=2,
            N=N,
            integration_domain=anp.array([[0, 1.0], [-2.0, 0.0]], like=backend),
        )
        assert anp.abs(result - (-4.0)) < err_max

    for integr, N in zip(integrators_nc, Ns_nc):
        print(
            f"[2mTesting {type(integr).__name__} with {backend} and a linear"
            " function[m"
        )
        result = integr.integrate(
            fn=fn_linear,
            dim=2,
            N=N,
            integration_domain=anp.array([[0, 1.0], [-2.0, 0.0]], like=backend),
        )
        assert anp.abs(result - 1.0) < err_max
    print(f"Tests passed for backend {backend}.")


def test_integrate_double():
    """
    Set the precision to double for backends where possible and call the
    run_simple_integrations for all supported backends.
    """
    enable_cuda()
    set_precision("double", backend="torch")
    set_precision("double", backend="jax")
    # Numpy uses double and tensorflow uses float; these cannot (yet) be changed
    # globally

    run_simple_integrations("torch")
    run_simple_integrations("numpy")
    run_simple_integrations("jax")


def test_integrate_float():
    """
    Set the precision to float for backends where possible and call the
    run_simple_integrations for all supported backends.
    """
    enable_cuda()
    set_precision("float", backend="torch")
    set_precision("float", backend="jax")

    run_simple_integrations("torch", err_max=1e-5, dtype_name="float32")
    run_simple_integrations("jax", err_max=1e-5, dtype_name="float32")
    run_simple_integrations("tensorflow", err_max=1e-5, dtype_name="float32")


if __name__ == "__main__":
    try:
        test_integrate_double()
        test_integrate_float()
    except KeyboardInterrupt:
        pass
