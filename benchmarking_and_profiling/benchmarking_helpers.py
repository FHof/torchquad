import sys

sys.path.append("..")

import numpy as np
import time
import timeit
import traceback
import csv

from torchquad import set_log_level


def measure_time(func, mode, num_runs, warmups=3):
    """
    Measure the median run-time in s or a similar statistical value

    Args:
        func (function): The function which should be measured
        mode (string): The type of timing mechanism
        num_runs (int): Number of function executions to calculate statistics
        warmups (int): Number of warmup executions before taking measurements
    """
    if mode == "perf_counter":
        for _ in range(warmups):
            func()
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            func()
            times.append(time.perf_counter() - t0)
        return np.median(times)
    elif mode == "timeit":
        # https://docs.python.org/3/library/timeit.html
        for _ in range(warmups):
            func()
        times = timeit.Timer(stmt="func()", globals={"func": func}).repeat(
            repeat=num_runs, number=1
        )
        times_sum = sum(times)
        if times_sum < 0.2:
            # Do extra runs if the function is fast enough
            time_per_run = times_sum / num_runs
            num_extra_runs = min(int((0.2 - times_sum) / time_per_run), 400)
            times = times + timeit.Timer(stmt="func()", globals={"func": func}).repeat(
                repeat=num_extra_runs, number=1
            )
        print(f"Taking median from {len(times)} runs")
        return np.median(times)
    elif mode == "pytorch_benchmark":
        # https://pytorch.org/docs/stable/benchmark_utils.html
        # https://github.com/pytorch/pytorch/tree/master/torch/utils/benchmark
        # https://pytorch.org/tutorials/recipes/recipes/benchmark.html
        import torch.utils.benchmark as benchmark
        import torch

        # repeat is not yet implemented, so use blocked_autorange.
        measurement = benchmark.Timer(
            stmt="func()",
            globals={"func": func},
            num_threads=torch.get_num_threads(),
        ).blocked_autorange(min_run_time=2.0)
        # ~ ).repeat(repeat=num_runs, number=1)
        # ~ if measurement.number_per_run > 1 or len(measurement.times) < num_runs:
        print(f"Taking 'median' from {measurement}")
        return measurement.median


def get_Ns(dim, min_N, max_N, integrator_name, base):
    """
    Calculate a list of valid number of points for the given integrator
    which can be used to compare runtimes.
    The number of points is roughly defined by the sequence (base ^ k)_k
    """
    Ns = []
    N_desired = 1.0
    for _ in range(1000):
        N = -1
        if integrator_name in ["MonteCarlo", "VEGAS"]:
            N = int(N_desired)
            if N < 5:
                # MonteCarlo works badly with very few points
                N = 5
        elif integrator_name == "Boole":
            n_per_dim = int(N_desired ** (1.0 / dim) + 1e-8)
            if n_per_dim < 5:
                n_per_dim = 5
            new_n_per_dim = n_per_dim - ((n_per_dim - 1) % 4)
            N = new_n_per_dim**dim
        elif integrator_name == "Simpson":
            n_per_dim = int(N_desired ** (1.0 / dim) + 1e-8)
            if n_per_dim < 3:
                n_per_dim = 3
            new_n_per_dim = n_per_dim - ((n_per_dim - 1) % 2)
            N = new_n_per_dim**dim
        elif integrator_name == "Trapezoid":
            n_per_dim = int(N_desired ** (1.0 / dim) + 1e-8)
            if n_per_dim < 2:
                n_per_dim = 2
            N = n_per_dim**dim
        if N > max_N:
            break
        if (len(Ns) == 0 or Ns[-1] < N) and N >= min_N:
            Ns.append(N)
        # Increase N roughly exponentially with the given base
        N_desired = N_desired * base
    return Ns


def measure_runtimes(
    integrator_name, dim, run_func, backend, min_N, max_N, max_median_time, base
):
    """
    Measure the times it takes to integrate for multiple values of N

    Args:
        integrator_name (string): The integrator name, e.g. "Trapezoid"
        dim (int): Dimensionality
        run_func (function(N)): A function which executes integration or a similar operation with a given N
        backend (string): Numerical backend
        min_N (int): Minimum number of points
        max_N (int): Maximum number of points
        base (float): base argument passed to get_Ns
    """
    set_log_level("SUCCESS")

    num_runs = 11
    warmups = 2
    measuring_mode = {
        "torch": "pytorch_benchmark",
        "numpy": "timeit",
        "jax": "timeit",
        "tensorflow": "timeit",
    }[backend]

    measurement_data = []
    for N in get_Ns(dim, min_N, max_N, integrator_name, base=base):
        print(f"Measuring times for integrator: {integrator_name}, N: {N}")
        median_runtime = None
        # Abort when an exception, e.g. out of memory, happens so that
        # previous measurements may nonetheless be saved
        try:
            median_runtime = measure_time(
                lambda: run_func(N),
                measuring_mode,
                num_runs=num_runs,
                warmups=warmups,
            )
        except Exception as err:
            traceback.print_exc()
            print(f"Aborting measurements because of failure: {err}")
            break
        measurement_data.append(
            {
                "median_runtime": median_runtime,
                "N": N,
            }
        )
        if median_runtime > max_median_time:
            break
    set_log_level("INFO")
    return measurement_data


def save_measurement_data(measurement_data, output_file_path):
    """Append the measurement results to a CSV file."""
    if len(measurement_data) == 0:
        print("Nothing to save")
        return
    print(f"Saving {len(measurement_data)} measurements to {output_file_path}")
    with open(output_file_path, "a") as csv_file:
        writer = csv.DictWriter(csv_file, measurement_data[0].keys())
        if csv_file.tell() == 0:
            writer.writeheader()
        for measurement_dict in measurement_data:
            writer.writerow(measurement_dict)
