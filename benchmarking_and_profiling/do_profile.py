#!/usr/bin/env python3

import sys

sys.path.append("../../torchquad")

import argparse
from autoray import numpy as anp
from autoray import to_numpy, infer_backend, to_backend_dtype
import cProfile
from pathlib import Path

from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS, RNG
from common_helpers import setup_for_backend, integrand_functions


def run_with_cprofile(func, output_file):
    with cProfile.Profile() as pr:
        func()
    pr.dump_stats(output_file)
    pr.print_stats("tottime")


def wrap_integrand(integrand, backend, do_block=True):
    """
    Change the integrand to wait until its inputs and outputs are calculated
    if possible and add a trace annotation.
    """
    new_integrand = None
    if backend == "jax":
        import jax

        if do_block:

            def new_integrand(x):
                x = x.block_until_ready()
                with jax.profiler.TraceAnnotation("integrand execution (synced)"):
                    return integrand(x).block_until_ready()

        else:

            def new_integrand(x):
                with jax.profiler.TraceAnnotation("integrand execution"):
                    return integrand(x)

    elif backend == "torch":
        import torch

        if torch.cuda.is_available() and do_block:

            def new_integrand(x):
                torch.cuda.synchronize()
                with torch.profiler.record_function("integrand evaluation (synced)"):
                    result = integrand(x)
                    torch.cuda.synchronize()
                    return result

        else:

            def new_integrand(x):
                with torch.profiler.record_function("integrand evaluation"):
                    return integrand(x)

    elif backend == "tensorflow":
        import tensorflow as tf

        def new_integrand(x):
            # Add a Trace Event for the integrand execution time
            with tf.profiler.experimental.Trace("integrand evaluation"):
                return integrand(x)

    elif backend == "numpy":
        new_integrand = integrand
    return new_integrand


def get_integrate_func(
    compile_all,
    compile_parts,
    calculate_gradient,
    integrator,
    integration_domain,
    integrand,
    N,
):
    """Get a function which executes a possibly compiled integrate method with the given arguments"""
    assert (compile_all, compile_parts) != (True, True)
    backend = infer_backend(integration_domain)
    assert backend in ["numpy", "torch", "jax", "tensorflow"]
    dim = integration_domain.shape[0]

    run_func = None
    if backend == "jax":
        import jax

        if compile_parts:
            integrate_compiled = integrator.get_jit_compiled_integrate(
                dim, N, integration_domain
            )

            if calculate_gradient:
                integrand = wrap_integrand(jax.jit(integrand), backend, do_block=False)
                grad_integrate = jax.grad(
                    lambda dom: integrate_compiled(integrand, dom)
                )

                def run_func():
                    return to_numpy(
                        grad_integrate(integration_domain).block_until_ready()
                    )

            else:
                integrand = wrap_integrand(jax.jit(integrand), backend)

                def run_func():
                    return to_numpy(
                        integrate_compiled(
                            integrand, integration_domain
                        ).block_until_ready()
                    )

        else:
            is_montecarlo = isinstance(integrator, MonteCarlo)
            if is_montecarlo:
                extra_kwargs = {"rng": RNG(backend="jax", seed=None)}
            else:
                extra_kwargs = {}
            integrate = integrator.integrate
            if calculate_gradient:
                # block_until_ready does not work on ConcreteArray objects
                integrand = wrap_integrand(integrand, backend, do_block=False)
                grad_integrate = jax.grad(
                    lambda dom: integrate(
                        integrand, N=N, integration_domain=dom, dim=dim, **extra_kwargs
                    )
                )
                if compile_all:
                    if is_montecarlo:
                        rng = extra_kwargs["rng"]
                        rng_key = rng.jax_get_key()
                        grad_integrate_uncompiled = grad_integrate

                        @jax.jit
                        def jit_grad_integrate(dom, rng_key):
                            rng.jax_set_key(rng_key)
                            gradient = grad_integrate_uncompiled(dom)
                            return gradient, rng.jax_get_key()

                        def grad_integrate(dom):
                            nonlocal rng_key
                            gradient, rng_key = jit_grad_integrate(dom, rng_key)
                            return gradient

                    else:
                        grad_integrate = jax.jit(grad_integrate)

                def run_func():
                    return to_numpy(
                        grad_integrate(integration_domain).block_until_ready()
                    )

            else:
                if compile_all:
                    integrand = wrap_integrand(integrand, backend, do_block=False)
                    if is_montecarlo:
                        rng = extra_kwargs["rng"]
                        rng_key = rng.jax_get_key()
                        integrate_uncompiled = integrate

                        @jax.jit
                        def jit_integrate(dom, rng_key):
                            rng.jax_set_key(rng_key)
                            result = integrate_uncompiled(
                                integrand,
                                N=N,
                                integration_domain=dom,
                                dim=dim,
                                **extra_kwargs,
                            )
                            return result, rng.jax_get_key()

                        def integrate(fn, N, integration_domain, dim):
                            nonlocal rng_key
                            result, rng_key = jit_integrate(integration_domain, rng_key)
                            return result

                    else:
                        integrate = jax.jit(
                            integrate, static_argnames=["fn", "N", "dim", "backend"]
                        )
                else:
                    integrand = wrap_integrand(integrand, backend)

                def run_func():
                    return to_numpy(
                        integrate(
                            integrand,
                            N=N,
                            integration_domain=integration_domain,
                            dim=dim,
                        ).block_until_ready()
                    )

    elif backend == "tensorflow":
        import tensorflow as tf

        def tf_compile(func):
            return tf.function(func, jit_compile=True)

        if isinstance(integrator, MonteCarlo) and compile_all:
            extra_kwargs = {"rng": RNG(backend="tensorflow", seed=None)}
        else:
            extra_kwargs = {}

        if compile_parts:
            integrand = wrap_integrand(tf_compile(integrand), backend)
            integrate_compiled = integrator.get_jit_compiled_integrate(
                dim, N, integration_domain, **extra_kwargs
            )

            if calculate_gradient:

                def run_func():
                    dom = tf.Variable(integration_domain)
                    with tf.GradientTape() as tape:
                        result = integrate_compiled(integrand, dom)
                    gradient = tape.gradient(result, dom)
                    return to_numpy(gradient)

            else:

                def run_func():
                    return to_numpy(integrate_compiled(integrand, integration_domain))

        else:
            integrate = integrator.integrate
            if compile_all:
                integrand = wrap_integrand(integrand, backend, do_block=False)
                integrate = tf_compile(integrate)
            else:
                integrand = wrap_integrand(integrand, backend)

            if calculate_gradient:

                def run_func():
                    dom = tf.Variable(integration_domain)
                    with tf.GradientTape() as tape:
                        result = integrate(
                            integrand,
                            N=N,
                            integration_domain=dom,
                            dim=dim,
                            **extra_kwargs,
                        )
                    gradient = tape.gradient(result, dom)
                    return to_numpy(gradient)

            else:

                def run_func():
                    return to_numpy(
                        integrate(
                            integrand,
                            N=N,
                            integration_domain=integration_domain,
                            dim=dim,
                            **extra_kwargs,
                        )
                    )

    elif backend == "torch":
        import torch

        if calculate_gradient:
            integration_domain = integration_domain.detach()
            integration_domain.requires_grad = True
        if compile_all:
            # Add synchronisation and trace info to the integrand;
            # this may not work because it's included in the compilation
            integrand = wrap_integrand(integrand, backend)

            # Define a traceable integrate function, compile it and use it
            # for the to-be-profiled function
            def func_compilable(integration_domain):
                return integrator.integrate(
                    fn=integrand,
                    N=N,
                    integration_domain=integration_domain,
                    dim=dim,
                )

            check_trace = not isinstance(integrator, MonteCarlo)
            integrate_compiled = torch.jit.trace(
                func_compilable, (integration_domain,), check_trace=check_trace
            )

            if calculate_gradient:

                def run_func():
                    dom = integration_domain.detach()
                    dom.requires_grad = True
                    result = integrate_compiled(dom)
                    assert hasattr(result, "grad_fn")
                    result.backward()
                    gradient = dom.grad
                    return to_numpy(gradient)

            else:

                def run_func():
                    return to_numpy(integrate_compiled(integration_domain))

        elif compile_parts:
            # Get example input for the integrand and use it to compile the
            # integrand
            if isinstance(integrator, MonteCarlo):
                sample_points = integrator.calculate_sample_points(
                    N, integration_domain, seed=0
                )
            else:
                sample_points, _, _ = integrator.calculate_grid(N, integration_domain)
            if calculate_gradient:
                # Avoid the warning about a .grad attribute access of a non-leaf
                # Tensor
                sample_points = sample_points.detach()
                sample_points.requires_grad = True
            integrand = torch.jit.trace(integrand, (sample_points,))
            # Add synchronisation and trace info to the compiled integrand
            integrand = wrap_integrand(integrand, backend)
            # Compile the first and last step of the integrate function
            integrate_compiled = integrator.get_jit_compiled_integrate(
                dim, N, integration_domain
            )

            if calculate_gradient:

                def run_func():
                    dom = integration_domain.detach()
                    dom.requires_grad = True
                    result = integrate_compiled(integrand, dom)
                    assert hasattr(result, "grad_fn")
                    result.backward()
                    gradient = dom.grad
                    return to_numpy(gradient)

            else:

                def run_func():
                    return to_numpy(integrate_compiled(integrand, integration_domain))

        else:
            integrand = wrap_integrand(integrand, backend)

            if calculate_gradient:

                def run_func():
                    dom = integration_domain.detach()
                    dom.requires_grad = True
                    result = integrator.integrate(
                        fn=integrand,
                        N=N,
                        integration_domain=dom,
                        dim=dim,
                    )
                    result.backward()
                    gradient = dom.grad
                    return to_numpy(gradient)

            else:

                def run_func():
                    return to_numpy(
                        integrator.integrate(
                            fn=integrand,
                            N=N,
                            integration_domain=integration_domain,
                            dim=dim,
                        )
                    )

        if torch.cuda.is_available():
            # Add synchronisation before and after the function
            run_func_unsynced = run_func

            def run_func():
                torch.cuda.synchronize()
                result = run_func_unsynced()
                torch.cuda.synchronize()
                return result

    elif backend == "numpy":
        assert (compile_all, compile_parts, calculate_gradient) == (False, False, False)

        def run_func():
            return integrator.integrate(
                fn=integrand,
                N=N,
                integration_domain=integration_domain,
                dim=dim,
            )

    return run_func


def profile_backend_specific(backend, run_func, output_folder, torch_use_pyprof):
    """Profile run_func with a backend-specific profiler"""
    if backend == "jax":
        import jax

        with jax.profiler.trace(str(output_folder.joinpath("jax_tensorboard"))):
            run_func()
        jax.profiler.save_device_memory_profile(
            str(output_folder.joinpath("jax_memory.prof"))
        )

    elif backend == "tensorflow":
        import tensorflow as tf

        options = tf.profiler.experimental.ProfilerOptions(python_tracer_level=1)
        with tf.profiler.experimental.Profile(
            str(output_folder.joinpath("tf_tensorboard")), options=options
        ):
            run_func()

    elif backend == "torch":
        import torch

        if not torch_use_pyprof:
            with torch.profiler.profile(
                profile_memory=True,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(output_folder.joinpath("torch_tensorboard"))
                ),
            ) as p:
                run_func()
            print(p.key_averages().table(sort_by="cpu_time", row_limit=-1))
        else:
            import torch.cuda.profiler as cuda_profiler
            import pyprof

            pyprof.init()
            with torch.autograd.profiler.emit_nvtx():
                # FIXME: What is the exact purpose of
                # cuda_profiler.start() and cuda_profiler.stop()?
                cuda_profiler.start()
                run_func()
                cuda_profiler.stop()

    elif backend == "numpy":
        pass


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
        "precision",
        help="Floating point precision",
        choices=["float32", "float64"],
    )
    parser.add_argument(
        "integrator",
        help="Integrator name",
        choices=["Trapezoid", "Boole", "Simpson", "MonteCarlo", "VEGAS"],
    )
    parser.add_argument(
        "--use-cprofile",
        action="store_true",
        help="Use cProfile instead of backend-specific profiling",
    )
    parser.add_argument(
        "--N",
        help="Number of evaluation points",
        default=17850625,
        type=int,
    )
    parser.add_argument(
        "--use-pyprof",
        action="store_true",
        help="Use PyProf for Torch instead of torch.profiler",
    )
    parser.add_argument(
        "--output-folder",
        help="Output folder for the collected data",
        default="out_profiled",
    )
    compile_arg_group = parser.add_mutually_exclusive_group()
    compile_arg_group.add_argument(
        "--compile-all",
        action="store_true",
        help="Compile the whole integrate method before profiling it",
    )
    compile_arg_group.add_argument(
        "--compile-parts",
        action="store_true",
        help="Compile the three steps of integration individually",
    )
    parser.add_argument(
        "--calculate-gradient",
        action="store_true",
        help="Calculate the gradient with respect to the integration domain",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    backend = args.backend
    dtype_name = args.precision
    setup_for_backend(backend, dtype_name, use_tpu=False)
    dim = 4
    integration_domain = [[0.0, 1.0]] * dim
    integrator = {
        "Boole": Boole,
        "Trapezoid": Trapezoid,
        "Simpson": Simpson,
        "MonteCarlo": MonteCarlo,
        "VEGAS": VEGAS,
    }[args.integrator]()
    N = args.N
    use_cprofile = args.use_cprofile
    compile_parts = args.compile_parts
    compile_all = args.compile_all
    calculate_gradient = args.calculate_gradient
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    integrand = integrand_functions["sin_prod"]

    # Convert integration_domain to a backend-specific tensor
    integration_domain = anp.array(
        integration_domain,
        like=backend,
        dtype=to_backend_dtype(dtype_name, like=backend),
    )

    # Get a function to be profiled
    run_func = get_integrate_func(
        compile_all,
        compile_parts,
        calculate_gradient,
        integrator,
        integration_domain,
        integrand,
        N,
    )

    # Execute a few warmups to ensure that compilation has happened
    for _ in range(2):
        print(f"output from warmup: {run_func()}")

    if not use_cprofile:
        profile_backend_specific(backend, run_func, output_folder, args.use_pyprof)
    else:
        # Common code for cProfile
        run_with_cprofile(
            run_func, str(output_folder.joinpath(f"{backend}_cprofile_output.prof"))
        )


main()
