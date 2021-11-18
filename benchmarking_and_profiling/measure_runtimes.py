#!/usr/bin/env python3

import sys

sys.path.append("..")

import argparse
from autoray import numpy as anp
from autoray import to_numpy, to_backend_dtype, infer_backend

from torchquad import Trapezoid, Simpson, Boole, MonteCarlo
from torchquad import RNG
from benchmarking_helpers import measure_runtimes, save_measurement_data
from common_helpers import setup_for_backend, integrand_functions


def block_integrand(integrand, backend):
    """
    Change the integrand to wait until its inputs and outputs are calculated
    if possible
    """
    print("Blocking an integrand function")
    new_integrand = None
    if backend == "jax":

        def new_integrand(x):
            return integrand(x.block_until_ready()).block_until_ready()

    elif backend == "torch":
        import torch

        if torch.cuda.is_available():

            def new_integrand(x):
                torch.cuda.synchronize()
                result = integrand(x)
                torch.cuda.synchronize()
                return result

        else:
            new_integrand = integrand
            print("No torch synchronization for CPU")
    elif backend == "tensorflow":
        new_integrand = integrand
        print("No TensorFlow synchronization")
    elif backend == "numpy":
        new_integrand = integrand
    return new_integrand


def compile_integrate_parts_torch(
    N, integrator, integration_domain, integrand, compile_integrand, sync_integrand
):
    """
    Compile three steps of the integrate method for torch and assemble a partly compiled new integrate function

    Args:
        N (int): Number of points
        integrator (NewtonCotes subclass): Integrator object
        integration_domain (torch tensor): Example input for integration_domain
        integrand (function): Integrand used to calculate example outputs. The integrand is needed if its output has a special dtype, e.g. complex numbers, or if it should be compiled.
        compile_integrand (Bool): If True, compile the integrand function
        sync_integrand (Bool): If True, add torch.cuda.synchronize before and after the integrand evaluation

    Returns:
        function(fn, integration_domain): Partly compiled integrate function
        function(x): Changed, e.g. compiled, or unchanged integrand, depending on the arguments
    """
    import torch

    # Get the partially compiled integrate function
    dim = integration_domain.shape[0]
    integrate_compiled = integrator.get_jit_compiled_integrate(
        dim, N, integration_domain
    )

    if compile_integrand:
        # Get example input for the integrand and use it to compile the
        # integrand
        if isinstance(integrator, MonteCarlo):
            sample_points = integrator.calculate_sample_points(N, integration_domain)
            integrand = torch.jit.trace(integrand, (sample_points,))
        else:
            grid_points, hs, n_per_dim = integrator.calculate_grid(
                N, integration_domain
            )
            n_per_dim = int(n_per_dim)
            integrand = torch.jit.trace(integrand, (grid_points,))

    if sync_integrand:
        # Add synchronisation to the compiled integrand
        integrand = block_integrand(integrand, "torch")

    return integrate_compiled, integrand


def compile_integrate(integrator, integrand, integration_domain):
    """Compile the whole integrate method

    Args:
        integrator: Integrator object
        integrand (function): Function which should be integrated
        integration_domain (backend tensor): Integration domain

    Returns:
        function(N): Function which executes a compiled version of integrate on integration_domain; if N changes, recompilation happens
    """
    backend = infer_backend(integration_domain)
    dim = integration_domain.shape[0]

    if backend == "jax":
        import jax

        jitted_integrate = jax.jit(
            integrator.integrate, static_argnames=["fn", "N", "dim", "backend"]
        )

        def run_integration(N):
            return jitted_integrate(
                fn=integrand,
                N=N,
                integration_domain=integration_domain,
                dim=dim,
                backend=backend,
            )

    elif backend == "tensorflow":
        import tensorflow as tf

        jitted_integrate = tf.function(integrator.integrate, jit_compile=True)

        extra_args = {}
        if isinstance(integrator, MonteCarlo):
            extra_args["rng"] = RNG(backend=backend)

        def run_integration(N):
            return jitted_integrate(
                fn=integrand,
                N=N,
                integration_domain=integration_domain,
                dim=dim,
                backend=backend,
                **extra_args,
            )

    elif backend == "torch":
        import torch

        compiled = [None, None]

        def do_compile(N):
            def func_compilable(integration_domain):
                return integrator.integrate(
                    fn=integrand,
                    N=N,
                    integration_domain=integration_domain,
                    dim=dim,
                    backend=backend,
                )

            check_trace = not isinstance(integrator, MonteCarlo)
            integrate_compiled = torch.jit.trace(
                func_compilable, (integration_domain,), check_trace=check_trace
            )
            return integrate_compiled

        def run_integration(N):
            if compiled[0] != N:
                # Compile on the first execution
                compiled[:] = (N, do_compile(N))
            return compiled[1](integration_domain)

    else:
        raise ValueError("unsupported backend for compilation")

    # Execute to_numpy on the integral result to force it to CPU and ensure all
    # computation happened
    return lambda N: to_numpy(run_integration(N))


def compile_integrate_grad(integrator, integrand, integration_domain):
    """Compile the whole integrate method and use it to calculate the gradient over integration_domain"""
    backend = infer_backend(integration_domain)
    dim = integration_domain.shape[0]

    if backend == "jax":
        import jax

        compiled = [None, None]

        def run_func(N):
            if compiled[0] != N:

                def func(integration_domain):
                    return integrator.integrate(
                        fn=integrand,
                        N=N,
                        integration_domain=integration_domain,
                        dim=dim,
                        backend=backend,
                    )

                grad_func = jax.grad(func)
                compiled[:] = (N, jax.jit(grad_func))
            return compiled[1](integration_domain)

    elif backend == "tensorflow":
        import tensorflow as tf

        extra_args = {}
        if isinstance(integrator, MonteCarlo):
            extra_args["rng"] = RNG(backend=backend)

        def integrate_grad(N, domain):
            with tf.GradientTape() as tape:
                result = integrator.integrate(
                    fn=integrand,
                    N=N,
                    integration_domain=domain,
                    dim=dim,
                    backend=backend,
                    **extra_args,
                )
            return tape.gradient(result, domain)

        jitted_integrate_grad = tf.function(integrate_grad, jit_compile=True)

        def run_func(N):
            domain = tf.Variable(integration_domain)
            return jitted_integrate_grad(N, domain)

    elif backend == "torch":
        import torch

        compiled = [None, None]

        def do_compile(N):
            # Including the backward step in the compilation did not work:
            # Cannot insert a Tensor that requires grad as a constant.
            def func_compilable(integration_domain):
                return integrator.integrate(
                    fn=integrand,
                    N=N,
                    integration_domain=integration_domain,
                    dim=dim,
                    backend=backend,
                )

            check_trace = not isinstance(integrator, MonteCarlo)
            domain = integration_domain.clone()
            domain.requires_grad = True
            integrate_compiled = torch.jit.trace(
                func_compilable, (domain,), check_trace=check_trace
            )
            return integrate_compiled

        def run_func(N):
            if compiled[0] != N:
                # Compile on the first execution
                compiled[:] = (N, do_compile(N))
            domain = integration_domain.clone()
            domain.requires_grad = True
            result = compiled[1](domain)
            result.backward()
            return domain.grad

    else:
        raise ValueError("unsupported backend for compilation")

    # Execute to_numpy on the integral result to force it to CPU and ensure all
    # computation happened
    return lambda N: to_numpy(run_func(N))


def compile_parts(integrator, integrand, integration_domain, sync_integrand):
    """Compile the three steps of integrate separately

    Args:
        integrator: Integrator object
        integrand (function): Function which should be integrated
        integration_domain (backend tensor): Integration domain
        sync_integrand (Bool): If True, wait until the execution finishes before and after the integrand evaluation

    Returns:
        function(N): Function which executes a partially compiled version of integrate on integration_domain; if N changes, recompilation happens
    """
    backend = infer_backend(integration_domain)
    dim = integration_domain.shape[0]

    if backend == "jax":
        import jax

        integrand = jax.jit(integrand)
        if sync_integrand:
            integrand = block_integrand(integrand, backend)
        compiled = [None, None]

        def run_integration(N):
            if compiled[0] != N:
                compiled[:] = (
                    N,
                    integrator.get_jit_compiled_integrate(dim, N, integration_domain),
                )
            return compiled[1](integrand, integration_domain)

    elif backend == "tensorflow":
        import tensorflow as tf

        def tf_compile(func):
            return tf.function(func, jit_compile=True)

        integrand = tf_compile(integrand)
        if sync_integrand:
            integrand = block_integrand(integrand, backend)
        compiled = [None, None]

        def run_integration(N):
            if compiled[0] != N:
                compiled[:] = (
                    N,
                    integrator.get_jit_compiled_integrate(dim, N, integration_domain),
                )
            return compiled[1](integrand, integration_domain)

    elif backend == "torch":
        # A list to remember the compiled functions
        compiled_funcs = [-1, None, None]

        def run_integration(N):
            if compiled_funcs[0] != N:
                # Compile the first and third integrate step and the
                # integrand in the first warmup iteration
                (
                    integrate_compiled,
                    integrand_compiled,
                ) = compile_integrate_parts_torch(
                    N,
                    integrator,
                    integration_domain,
                    integrand,
                    compile_integrand=True,
                    sync_integrand=sync_integrand,
                )
                # Remember the compiled functions for the next iterations
                compiled_funcs[:] = (N, integrate_compiled, integrand_compiled)

            # Execute the compiled functions
            _, integrate_compiled, integrand_compiled = compiled_funcs
            return integrate_compiled(integrand_compiled, integration_domain)

    else:
        raise ValueError("unsupported backend for parts compilation")

    # Execute to_numpy on the integral result to force it to CPU and ensure all
    # computation happened
    return lambda N: to_numpy(run_integration(N))


def compile_parts_grad(integrator, integrand, integration_domain, sync_integrand):
    """Compile the three steps of integrate separately and calculate a gradient over the integration_domain"""
    backend = infer_backend(integration_domain)
    dim = integration_domain.shape[0]

    if backend == "jax":
        import jax

        integrand = jax.jit(integrand)
        if sync_integrand:
            integrand = block_integrand(integrand, backend)
        compiled = [None, None]

        def run_integration(N):
            if compiled[0] != N:
                jit_integrate = integrator.get_jit_compiled_integrate(
                    dim, N, integration_domain
                )
                compiled[:] = (
                    N,
                    jax.grad(lambda domain: jit_integrate(integrand, domain)),
                )
            return compiled[1](integration_domain)

    elif backend == "tensorflow":
        import tensorflow as tf

        def tf_compile(func):
            return tf.function(func, jit_compile=True)

        integrand = tf_compile(integrand)
        if sync_integrand:
            integrand = block_integrand(integrand, backend)
        compiled = [None, None]

        def run_integration(N):
            domain = tf.Variable(integration_domain)
            if compiled[0] != N:
                compiled[:] = (N, integrator.get_jit_compiled_integrate(dim, N, domain))
            with tf.GradientTape() as tape:
                result = compiled[1](integrand, domain)
            return tape.gradient(result, domain)

    elif backend == "torch":
        # A list to remember the compiled functions
        compiled_funcs = [-1, None, None]

        def run_integration(N):
            domain = integration_domain.clone()
            domain.requires_grad = True
            if compiled_funcs[0] != N:
                # Compile the first and third integrate step and the
                # integrand in the first warmup iteration
                (
                    integrate_compiled,
                    integrand_compiled,
                ) = compile_integrate_parts_torch(
                    N,
                    integrator,
                    domain,
                    integrand,
                    compile_integrand=True,
                    sync_integrand=sync_integrand,
                )
                # Remember the compiled functions for the next iterations
                compiled_funcs[:] = (N, integrate_compiled, integrand_compiled)

            # Execute the compiled functions
            _, integrate_compiled, integrand_compiled = compiled_funcs
            result = integrate_compiled(integrand_compiled, domain)
            result.backward()
            return domain.grad

    else:
        raise ValueError("unsupported backend for parts compilation")

    # Execute to_numpy on the integral result to force it to CPU and ensure all
    # computation happened
    return lambda N: to_numpy(run_integration(N))


def uncompiled_grad(integrator, integrand, integration_domain):
    """Return a function to calculate the gradient over the integration domain and do no compilations"""
    backend = infer_backend(integration_domain)
    dim = integration_domain.shape[0]

    def run_integration_uncompiled(N, domain):
        return integrator.integrate(
            fn=integrand, N=N, integration_domain=domain, dim=dim, backend=backend
        )

    if backend == "jax":
        import jax

        grad = [None, None]

        def run_integration(N):
            if grad[0] != N:
                grad[:] = (
                    N,
                    jax.grad(lambda domain: run_integration_uncompiled(N, domain)),
                )
            return grad[1](integration_domain)

    elif backend == "tensorflow":
        import tensorflow as tf

        def run_integration(N):
            domain = tf.Variable(integration_domain)
            with tf.GradientTape() as tape:
                result = run_integration_uncompiled(N, domain)
            return tape.gradient(result, domain)

    elif backend == "torch":

        def run_integration(N):
            domain = integration_domain.clone()
            domain.requires_grad = True
            result = run_integration_uncompiled(N, domain)
            result.backward()
            return domain.grad

    else:
        raise ValueError("unsupported backend for gradient calculation")

    # Execute to_numpy on the integral result to force it to CPU and ensure all
    # computation happened
    return lambda N: to_numpy(run_integration(N))


def get_step1(integrator, integration_domain, do_compile):
    """Get a function for the calculation of evaluation points

    Args:
        integrator: Integrator object
        integration_domain (backend tensor): Integration domain
        do_compile (Bool): If True, compile the function with integration_domain as only variable argument

    Returns:
        function(N): Function which executes an evaluation point calculation; if N changes and do_compile is True, recompilation happens
    """
    backend = infer_backend(integration_domain)
    run_func = None
    if isinstance(integrator, MonteCarlo):
        raise ValueError("NYI")
    else:
        if backend in ["tensorflow", "jax"]:
            if backend == "tensorflow":
                import tensorflow as tf

                calc_grid = integrator.calculate_grid
                if do_compile:
                    calc_grid = tf.function(calc_grid, jit_compile=True)

                def run_func(N):
                    grid_points, hs, n_per_dim = calc_grid(N, integration_domain)
                    return grid_points, hs, int(n_per_dim)

            elif backend == "jax":
                import jax

                calc_grid = integrator.calculate_grid
                if do_compile:
                    calc_grid = jax.jit(calc_grid, static_argnames=["N"])

                def run_func(N):
                    grid_points, hs, n_per_dim = calc_grid(N, integration_domain)
                    return (
                        grid_points.block_until_ready(),
                        hs.block_until_ready(),
                        int(n_per_dim),
                    )

        elif backend == "torch":
            if do_compile:
                compiled = [None, None]

                def compile_step1(N):
                    import torch

                    def step1(integration_domain):
                        grid_points, hs, n_per_dim = integrator.calculate_grid(
                            N, integration_domain
                        )
                        return (
                            grid_points,
                            hs,
                            torch.Tensor([n_per_dim]),
                        )  # n_per_dim is constant

                    return torch.jit.trace(step1, (integration_domain,))

                def run_func(N):
                    if compiled[0] != N:
                        compiled[:] = (N, compile_step1(N))
                    grid_points, hs, n_per_dim = compiled[1](integration_domain)
                    return grid_points, hs, int(n_per_dim)

            else:

                def run_func(N):
                    grid_points, hs, n_per_dim = integrator.calculate_grid(
                        N, integration_domain
                    )
                    return grid_points, hs, int(n_per_dim)

        elif backend == "numpy":
            assert not do_compile

            def run_func(N):
                return integrator.calculate_grid(N, integration_domain)

    return run_func


def get_step3(integrator, integrand, integration_domain, do_compile):
    """Get a function for the calculation of the final integral given function evaluations

    Args:
        integrator: Integrator object
        integrand (function): Function which should be integrated
        integration_domain (backend tensor): Integration domain
        do_compile (Bool): If True, compile the function with integration_domain as only variable argument

    Returns:
        function(N): Function which executes a final integral calculation; if N changes, new input is calculated and recompilation can happen
    """
    backend = infer_backend(integration_domain)
    dim = integration_domain.shape[0]
    run_func = None
    if isinstance(integrator, MonteCarlo):
        raise ValueError("NYI")
    else:

        def get_step3_input(N):
            grid_points, hs, n_per_dim = integrator.calculate_grid(
                N, integration_domain
            )
            function_values, _ = integrator.evaluate_integrand(integrand, grid_points)
            return function_values, dim, n_per_dim, hs

        calc_result = integrator.calculate_result
        if backend == "tensorflow" and do_compile:
            import tensorflow as tf

            calc_result = tf.function(calc_result, jit_compile=True)
        elif backend == "jax" and do_compile:
            import jax

            calc_result = jax.jit(calc_result, static_argnames=["dim", "n_per_dim"])
        if not (backend == "torch" and do_compile):
            ninputs = [None, None]

            def run_func(N):
                if ninputs[0] != N:
                    ninputs[:] = (N, get_step3_input(N))
                return to_numpy(calc_result(*ninputs[1]))

        else:
            compiled_and_inputs = [None, None, None]

            def compile_step3(N, inputs):
                import torch

                function_values, dim, n_per_dim, hs = inputs

                def step3(function_values, hs):
                    return calc_result(function_values, dim, n_per_dim, hs)

                return torch.jit.trace(step3, (function_values, hs))

            def run_func(N):
                if compiled_and_inputs[0] != N:
                    inputs = get_step3_input(N)
                    step3_compiled = compile_step3(N, inputs)
                    compiled_and_inputs[:] = (N, step3_compiled, inputs)
                function_values, _, _, hs = compiled_and_inputs[2]
                return to_numpy(compiled_and_inputs[1](function_values, hs))

        if backend == "numpy":
            assert not do_compile

    return run_func


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "backend",
        help="Numerical backend to measure",
        choices=["numpy", "torch", "jax", "tensorflow"],
    )
    parser.add_argument(
        "integrator",
        help="Integrator name for which plots should be created",
        choices=["Trapezoid", "Boole", "Simpson", "MonteCarlo"],
    )
    parser.add_argument(
        "integrand",
        help="Integrand function",
        choices=integrand_functions.keys(),
    )
    parser.add_argument(
        "dim",
        help="Dimensionality for the integration_domain",
        type=int,
    )
    parser.add_argument(
        "output_file",
        help="CSV file path to save the measurements",
    )
    parser.add_argument(
        "--min-N",
        help="Minimum number of evaluations",
        default=101,
        type=int,
    )
    parser.add_argument(
        "--max-N",
        help="Maximum number of evaluations",
        default=10000000000,
        type=int,
    )
    parser.add_argument(
        "--base",
        help="Base for choosing the values of N: (base ^ k)_k",
        default=10.0,
        type=float,
    )
    parser.add_argument(
        "--max-median-time",
        help="Skip remaining N values when a median time is at least this value in seconds",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--precision-dtype",
        help="Floating point precision",
        choices=["float32", "float64", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument(
        "--tpu",
        action="store_true",
        help="Use a TPU (Tensor Processing Unit) as accelerator",
    )
    steps_arg_group = parser.add_mutually_exclusive_group()
    steps_arg_group.add_argument(
        "--step1",
        action="store_true",
        help="Measure only evaluation point calculation",
    )
    steps_arg_group.add_argument(
        "--step3",
        action="store_true",
        help="Measure only the result calculation from function evaluation results",
    )
    compile_arg_group = parser.add_mutually_exclusive_group()
    compile_arg_group.add_argument(
        "--compile",
        action="store_true",
        help="Compile the integrate method before measuring it, e.g. with jax.jit",
    )
    compile_arg_group.add_argument(
        "--compile-parts",
        action="store_true",
        help="Compile the three steps of integration individually",
    )
    parser.add_argument(
        "--sync-integrand",
        action="store_true",
        help="Block around the integrand execution. Only affects --compile-parts without gradients",
    )
    parser.add_argument(
        "--calculate-gradient",
        action="store_true",
        help="Calculate the gradient with respect to the integration domain",
    )
    args = parser.parse_args()
    if args.compile_parts and (args.step1 or args.step3):
        raise ValueError("--compile-parts and --step1 cannot be used together")
    if args.integrand == "gaussian_peaks" and args.dim != 2:
        raise ValueError("gaussian_peaks supports only dim=2")
    if args.backend == "numpy" and (args.compile or args.compile_parts):
        raise ValueError("Cannot compile numpy")
    if args.calculate_gradient and (args.step1 or args.step3):
        raise ValueError(
            "Gradient calculation and step1 or step3 cannot be used together"
        )
    return args


def main():
    args = parse_arguments()
    backend = args.backend
    setup_for_backend(backend, args.precision_dtype, args.tpu)
    integrator = {
        "Boole": Boole,
        "Trapezoid": Trapezoid,
        "Simpson": Simpson,
        "MonteCarlo": MonteCarlo,
    }[args.integrator]()

    dim = args.dim
    integrand = integrand_functions[args.integrand]
    if args.integrand == "gaussian_peaks":
        assert dim == 2
        integration_domain = [[0.0, 1.0], [0.0, 1.0]]
    else:
        integration_domain = [[0.0, 4.0]] * dim
    integration_domain = anp.array(
        integration_domain,
        like=backend,
        dtype=to_backend_dtype(args.precision_dtype, like=backend),
    )
    # compiled_info will be set to a string in
    # ["no", "integrate", "three_steps", "step1", "step3"]
    # and represents which parts of the integration were compiled
    compiled_info = None

    run_func = None

    def run_integration_uncompiled(N):
        # Execute to_numpy so that the result is put to the main memory
        # (needs testing)
        return to_numpy(
            integrator.integrate(
                fn=integrand,
                N=N,
                integration_domain=integration_domain,
                dim=dim,
                backend=backend,
            )
        )

    if not args.calculate_gradient:
        if args.step1:
            run_func = get_step1(integrator, integration_domain, args.compile)
            if args.compile:
                compiled_info = "step1"
            else:
                compiled_info = "no"
        elif args.step3:
            run_func = get_step3(
                integrator, integrand, integration_domain, args.compile
            )
            if args.compile:
                compiled_info = "step3"
            else:
                compiled_info = "no"
        elif args.compile:
            run_func = compile_integrate(integrator, integrand, integration_domain)
            compiled_info = "integrate"
        elif args.compile_parts:
            sync_integrand = args.sync_integrand
            run_func = compile_parts(
                integrator, integrand, integration_domain, sync_integrand=sync_integrand
            )
            compiled_info = "three_steps"
        else:
            run_func = run_integration_uncompiled
            compiled_info = "no"
    else:
        if args.compile:
            run_func = compile_integrate_grad(integrator, integrand, integration_domain)
            compiled_info = "integrate"
        elif args.compile_parts:
            run_func = compile_parts_grad(
                integrator, integrand, integration_domain, sync_integrand=False
            )
            compiled_info = "three_steps"
        else:
            run_func = uncompiled_grad(integrator, integrand, integration_domain)
            compiled_info = "no"

    measurement_data = measure_runtimes(
        type(integrator).__name__,
        dim=dim,
        run_func=run_func,
        backend=backend,
        max_N=args.max_N,
        min_N=args.min_N,
        max_median_time=args.max_median_time,
        base=args.base,
    )
    # Add additional information to the measurement dicts
    integrator_name = args.integrator
    extra_info = {
        "integrator_name": integrator_name,
        "dim": dim,
        "integrand": args.integrand,
        "backend": backend,
        "compiled_info": compiled_info,
        "precision_dtype": args.precision_dtype,
    }
    if args.step1:
        extra_info["integrator_name"] = integrator_name + "_step1"
        extra_info["integrand"] = "unused"
    elif args.step3:
        extra_info["integrator_name"] = integrator_name + "_step3"
    elif args.calculate_gradient:
        extra_info["integrand"] = extra_info["integrand"] + " (gradient)"
    any(map(lambda measurement: measurement.update(extra_info), measurement_data))

    print(f"Measurements: {measurement_data}")
    save_measurement_data(measurement_data, args.output_file)


main()
