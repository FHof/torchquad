#!/usr/bin/env python3
"""
Compare the accuracies of integrals calculated with a VEGAS implementation for
different number of function evaluations.
Note that many configurations for the VEGAS integrators are hard-coded even if
they affect the accuracies.
"""
import sys

sys.path.append("..")
# Use the newest version of vegasflow because VEGAS+ is not yet in the latest
# release 1.2.2
sys.path.append("../../vegasflow/src")

import argparse
from autoray import numpy as anp
import time
import traceback

from common_helpers import (
    setup_for_backend,
    integrand_functions,
    get_reference_integral,
)
from benchmarking_helpers import get_Ns, save_measurement_data


num_evals = 0


def integrand_count_evals(integrand):
    """Count the number of evaluations before the integrand evaluation"""

    def new_integrand(x):
        global num_evals
        num_evals += x.shape[0]
        return integrand(x)

    return new_integrand


def get_integration(implementation, N, integrand, integration_domain):
    """Get a function which integrates the given integrand with the specified VEGAS implementation

    Args:
        implementation (string): VEGAS implementation name
        N (int): Approximate overall number of function evaluations
        integrand (function): To-be-integrated function
        integration_domain (list): Domain for the integration; should be [0,1]^dim

    Returns:
        function(): A function which integrates with VEGAS and returns the result
    """
    dim = len(integration_domain)
    if implementation == "VegasFlow":
        from vegasflow import VegasFlowPlus, run_eager

        compile_integration = True
        compile_integrand = False
        run_eager(not compile_integration)
        # Apparently VegasFlow always uses integration domain [0, 1]^dim
        assert all(arr[0] == 0.0 and arr[1] == 1.0 for arr in integration_domain)
        num_iterations = 7
        vegas_instance = VegasFlowPlus(dim, N // num_iterations)
        vegas_instance.compile(integrand, compilable=compile_integrand)
        return lambda: vegas_instance.run_integration(num_iterations)[0]
    elif implementation == "torchquad":
        from torchquad import VEGAS, set_log_level

        setup_for_backend("torch", "float32", False)
        set_log_level("WARNING")
        integrator = VEGAS()
        domain = anp.array(integration_domain, like="torch")
        # The number of iterations cannot easily be configured with torchquad
        return lambda: integrator.integrate(integrand, dim, N, domain)
    elif implementation == "gplepage":
        import vegas

        f = vegas.batchintegrand(integrand)

        def run_integration():
            integrator = vegas.Integrator(integration_domain, neval=N // (5 + 7))
            # Adapt the grid
            integrator(f, nitn=5)
            # Final integration
            return integrator(f, nitn=7, alpha=0.1).mean

        return run_integration
    elif implementation == "MonteCarlo":
        # Actually not a VEGAS implementation, but helpful for comparison
        from torchquad import MonteCarlo, set_log_level

        setup_for_backend("torch", "float32", False)
        set_log_level("WARNING")
        integrator = MonteCarlo()
        domain = anp.array(integration_domain, like="torch")
        return lambda: integrator.integrate(integrand, dim, N, domain)
    else:
        raise RuntimeError("Invalid implementation arg")


def run_measurements(implementation, integrand_name, dim, max_duration):
    """Execute an integration with the given parameters for multiple N multiple times with VEGAS and a [0,1]^dim domain

    Args:
        implementation (string): VEGAS implementation name
        integrand_name (string): Integrand function key, e.g. "sin_prod"
        dim (int): Dimensionality
        max_duration (float): Maximum time a measurement can take before
            aborting, in seconds

    Returns:
        list of dicts: Collected errors, durations, etc.
    """
    integration_domain = [[0.0, 1.0]] * dim
    reference_solution = get_reference_integral(integrand_name, integration_domain)
    print(f"Reference solution: {reference_solution}")
    integrand = integrand_count_evals(integrand_functions[integrand_name])
    measurements = []
    for N in get_Ns(dim, 500, 500000000, "VEGAS", 1.2):
        integrate_func = get_integration(
            implementation, N, integrand, integration_domain
        )
        break_outer = False
        # Try multiple times with the same N; the first two are warmups
        for i in range(6):
            global num_evals
            num_evals = 0
            duration = None
            # Abort when an exception, e.g. out of memory, happens so that
            # previous measurements may nonetheless be saved
            try:
                t0 = time.perf_counter()
                integral = integrate_func()
                duration = time.perf_counter() - t0
            except Exception as err:
                traceback.print_exc()
                print(f"Aborting measurements because of failure: {err}")
                break_outer = True
                break
            if i < 2:
                # Ignore warmups
                continue
            m = {
                "time": duration,
                "num_evals": num_evals,
                "error_abs": abs(float(integral - reference_solution)),
                "error_rel": abs(
                    float((integral - reference_solution) / reference_solution)
                ),
                "dim": dim,
                "N": N,
                "integrand": integrand_name,
            }
            print(f"Measurement: {m}")
            measurements.append(m)
            break_outer = duration > max_duration
            if break_outer:
                break
        if break_outer:
            break
    return measurements


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "implementation",
        help="VEGAS implementation or MonteCarlo",
        choices=["VegasFlow", "torchquad", "gplepage", "MonteCarlo"],
    )
    # ~ parser.add_argument(
    # ~ "--precision",
    # ~ help="Floating point precision",
    # ~ choices=["float32", "float64"],
    # ~ default="float64",
    # ~ )
    parser.add_argument(
        "--integrand",
        help="Integrand function",
        choices=integrand_functions.keys(),
        default="vegas_peak",
    )
    parser.add_argument(
        "--dim", help="Dimensionality for the integration_domain", type=int, default=4
    )
    parser.add_argument(
        "--max-duration",
        help="Maximum time in s an integration can take before aborting the measurements",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--output-file",
        help="CSV file path to save the measurements",
        default="tmp_vegas_measurements.csv",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    measurements = run_measurements(
        args.implementation, args.integrand, args.dim, args.max_duration
    )
    any(map(lambda m: m.update({"implementation": args.implementation}), measurements))
    # ~ any(map(lambda m: m.update({"precision": args.precision}), measurements))
    save_measurement_data(measurements, args.output_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
