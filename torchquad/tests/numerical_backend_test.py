import sys

sys.path.append("../")

import torch
import numpy as np
from autoray import infer_backend

from integration.trapezoid import Trapezoid
from integration.simpson import Simpson
from integration.boole import Boole
from integration.monte_carlo import MonteCarlo
from utils.enable_cuda import enable_cuda
from utils.set_precision import set_precision
from utils.set_log_level import set_log_level


def test_integrate():
    """
    Test if the integrators work with different numerical backends.
    A simple 2D constant function and a linear 2D function are defined and
    integrated for supported numerical backends to check if integration
    succeeds with a correct result.
    """
    set_log_level("INFO")
    enable_cuda()
    set_precision("double")

    # Newton Cotes composite integrators (nc); they should always integrate a
    # polynomial of degree 1 exactly
    integrators_nc = [Trapezoid(), Simpson(), Boole()]
    Ns_nc = [13 ** 2, 13 ** 2, 13 ** 2]
    integrators_all = integrators_nc + [MonteCarlo()]
    Ns_all = Ns_nc + [100]

    # Torch
    def fn_torch_const(x):
        assert infer_backend(x) == "torch"
        return 0.0 * x[:, 0] + torch.tensor(-2.0)

    for integr, N in zip(integrators_all, Ns_all):
        print(f"Testing {type(integr).__name__} with torch and a constant function")
        result = integr.integrate(
            fn=fn_torch_const,
            dim=2,
            N=N,
            integration_domain=torch.tensor([[0, 1.0], [-2.0, 0.0]]),
        )
        assert torch.abs(result - (-4.0)) < 1e-14

    def fn_torch_linear(x):
        assert infer_backend(x) == "torch"
        return 3.0 * x[:, 0] + x[:, 1]

    for integr, N in zip(integrators_nc, Ns_nc):
        print(f"Testing {type(integr).__name__} with torch and a linear function")
        result = integr.integrate(
            fn=fn_torch_linear,
            dim=2,
            N=N,
            integration_domain=torch.tensor([[0, 1.0], [-2.0, 0.0]]),
        )
        assert torch.abs(result - 1.0) < 1e-15
    print(f"Torch backend tests passed.")

    # Numpy
    def fn_numpy_const(x):
        assert infer_backend(x) == "numpy"
        return 0.0 * x[:, 0] - 2.0

    for integr, N in zip(integrators_all, Ns_all):
        print(f"Testing {type(integr).__name__} with numpy and a constant function")
        result = integr.integrate(
            fn=fn_numpy_const,
            dim=2,
            N=N,
            integration_domain=np.array([[0, 1.0], [-2.0, 0.0]]),
        )
        assert np.abs(result - (-4.0)) < 1e-14

    def fn_numpy_linear(x):
        assert infer_backend(x) == "numpy"
        return 3.0 * x[:, 0] + x[:, 1]

    for integr, N in zip(integrators_nc, Ns_nc):
        print(f"Testing {type(integr).__name__} with numpy and a linear function")
        result = integr.integrate(
            fn=fn_numpy_linear,
            dim=2,
            N=N,
            integration_domain=np.array([[0, 1.0], [-2.0, 0.0]]),
        )
        assert np.abs(result - 1.0) < 1e-15
    print(f"Numpy backend tests passed.")


if __name__ == "__main__":
    try:
        test_integrate()
    except KeyboardInterrupt:
        pass
