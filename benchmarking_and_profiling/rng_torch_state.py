#!/usr/bin/env python3
"""Test the RNG if it saves and recovers the state for torch"""
import sys

sys.path.append("..")

import torch
from autoray import to_backend_dtype
import time

from torchquad import VEGAS
from torchquad import set_up_backend, RNG


def check_vegas_samplepoints():
    """Try with integrand which resets the seed"""

    def integrand(x):
        print(f"first points: {x[:5]}")
        torch.random.manual_seed(300)
        return x[:, 0] * 0.0 + 6.0

    vegas = VEGAS()
    integral = vegas.integrate(
        integrand,
        2,
        N=10000,
        integration_domain=[[0.0, 3.0]] * 2,
        seed=None,
    )
    print("integral: ", integral)


def check_rng_performance():
    """Measure how fast or slow the RNG is"""
    print("Testing RNG performance")
    dtype = to_backend_dtype("float32", like="torch")
    for save_state in [False, True]:
        rng = RNG("torch", torch_save_state=save_state)
        # Assumption: < 1024 points are too few to be of interest for speed
        # comparions and > 1024 are too many points, where the state overhead
        # could disappear
        # A warmup
        for _ in range(3):
            rng.uniform(size=(1024, 3), dtype=dtype)
        t0 = time.perf_counter()
        for _ in range(50000):
            rng.uniform(size=(1024, 3), dtype=dtype)
        duration = time.perf_counter() - t0
        print(f"save_state: {save_state}, duration: {duration}")


def main():
    # ~ set_log_level("DEBUG")
    set_up_backend("torch", data_type="float32")
    check_rng_performance()
    # ~ check_vegas_samplepoints()


main()
