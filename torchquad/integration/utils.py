"""
Utility functions for the integrator implementations including extensions for
autoray, which are registered when importing this file
"""
import sys
from pathlib import Path

# Change the path to import from the parent folder.
# A relative import currently does not work when executing the tests.
sys.path.append(str(Path(__file__).absolute().parent.parent))

from autoray import numpy as anp
from autoray import infer_backend, get_dtype_name, register_function
from functools import partial
from loguru import logger

# from ..utils.set_precision import _get_precision
from utils.set_precision import _get_precision


def _linspace_with_grads(start, stop, N, requires_grad):
    """Creates an equally spaced 1D grid while keeping gradients
    in regard to inputs.
    Args:
        start (backend tensor): Start point (inclusive).
        stop (backend tensor): End point (inclusive).
        N (int): Number of points.
        requires_grad (bool): Indicates if output should be recorded for backpropagation in Torch.
    Returns:
        backend tensor: Equally spaced 1D grid
    """
    # The requires_grad case is only needed for Torch.
    if requires_grad:
        # Create 0 to 1 spaced grid
        grid = anp.linspace(
            anp.array(0.0, like=start), anp.array(1.0, like=start), N, dtype=start.dtype
        )

        # Scale to desired range, thus keeping gradients
        grid *= stop - start
        grid += start

        return grid
    else:
        if infer_backend(start) == "tensorflow":
            # Tensorflow determines the dtype automatically and doesn't support
            # the dtype argument here
            return anp.linspace(start, stop, N)
        return anp.linspace(start, stop, N, dtype=start.dtype)


def _add_at_indices(target, indices, source, is_sorted=False):
    """
    Add source[i] to target at target[indices[i]] for each index i in-place.
    For example, with targets=[0,0,0] indices=[2,1,1,2] and source=[a,b,c,d],
    targets will be changed to [0,b+c,a+d].
    This function supports only numpy and torch.

    Args:
        target (backend tensor): Tensor to which the source values are added
        indices (int backend tensor): Indices into target for each value in source
        source (backend tensor): Values which are added to target
        is_sorted (bool, optional): Set this to True if indices is monotonically increasing to skip a redundant sorting step with the numpy backend. Defaults to False.
    """
    backend = infer_backend(target)
    if backend == "torch":
        target.scatter_add_(dim=0, index=indices, src=source)
    else:
        # Use indicator matrices to reduce the Python interpreter overhead
        # Based on VegasFlow's consume_array_into_indices function
        # https://github.com/N3PDF/vegasflow/blob/21209c928d07c00ae4f789d03b83e518621f174a/src/vegasflow/utils.py#L16
        if not is_sorted:
            # Sort the indices and corresponding source array
            sort_permutation = anp.argsort(indices)
            indices = indices[sort_permutation]
            source = source[sort_permutation]
        # Maximum number of columns for the indicator matrices.
        # A higher number leads to more redundant comparisons and higher memory
        # usage but reduces the Python interpreter overhead.
        max_indicator_width = 500
        zero = anp.array(0.0, dtype=target.dtype, like=backend)
        num_indices = indices.shape[0]
        for i1 in range(0, num_indices, max_indicator_width):
            # Create an indicator matrix for source indices in {i1, i1+1, …, i2-1}
            # and corresponding target array indices in {t1, t1+1, …, t2-1}.
            # All other target array indices are irrelevant: because the indices
            # array is sorted, all values in indices[i1:i2] are bound by t1 and t2.
            i2 = min(i1 + max_indicator_width, num_indices)
            t1, t2 = indices[i1], indices[i2 - 1] + 1
            target_indices = anp.arange(t1, t2, dtype=indices.dtype, like=backend)
            indicator = anp.equal(indices[i1:i2], target_indices.reshape([t2 - t1, 1]))
            # Create a matrix which is zero everywhere except at entries where
            # the corresponding value from source should be added to the
            # corresponding entry in target, sum these source values, and add
            # the resulting vector to target
            target[t1:t2] += anp.sum(anp.where(indicator, source[i1:i2], zero), axis=1)


def _setup_integration_domain(dim, integration_domain, backend):
    """Sets up the integration domain if unspecified by the user.
    Args:
        dim (int): Dimensionality of the integration domain.
        integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend if possible.
        backend (string): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain.
    Returns:
        backend tensor: Integration domain.
    """
    logger.debug("Setting up integration domain.")

    # If no integration_domain is specified, create [-1,1]^d bounds
    if integration_domain is None:
        integration_domain = [[-1.0, 1.0]] * dim

    # Convert integration_domain to a tensor if needed
    if infer_backend(integration_domain) == "builtins":
        # Cast all integration domain values to Python3 float because
        # some numerical backends create a tensor based on the Python3 types
        integration_domain = [
            [float(b) for b in bounds] for bounds in integration_domain
        ]
        dtype_arg = _get_precision(backend)
        if dtype_arg is not None:
            # For Numpy and Tensorflow there is no global dtype, so set the
            # configured default dtype here
            integration_domain = anp.array(
                integration_domain, like=backend, dtype=dtype_arg
            )
        else:
            integration_domain = anp.array(integration_domain, like=backend)

    if integration_domain.shape != (dim, 2):
        raise ValueError(
            "The integration domain has an unexpected shape. "
            f"Expected {(dim, 2)}, got {integration_domain.shape}"
        )
    return integration_domain


def _check_integration_domain(integration_domain):
    """
    Check if the integration domain has a valid shape and determine the dimension.

    Args:
        integration_domain (list or backend tensor): Integration domain, e.g. [[-1,1],[0,1]].
    Returns:
        int: Dimension represented by the domain
    """
    if infer_backend(integration_domain) == "builtins":
        dim = len(integration_domain)
        if dim < 1:
            raise ValueError("len(integration_domain) needs to be 1 or larger.")

        for bounds in integration_domain:
            if len(bounds) != 2:
                raise ValueError(
                    bounds,
                    " in ",
                    integration_domain,
                    " does not specify a valid integration bound.",
                )
            if bounds[0] > bounds[1]:
                raise ValueError(
                    bounds,
                    " in ",
                    integration_domain,
                    " does not specify a valid integration bound.",
                )
        return dim
    else:
        if len(integration_domain.shape) != 2:
            raise ValueError("The integration_domain tensor has an invalid shape")
        dim, num_bounds = integration_domain.shape
        if dim < 1:
            raise ValueError("integration_domain.shape[0] needs to be 1 or larger.")
        if num_bounds != 2:
            raise ValueError("integration_domain must have 2 values per boundary")
        # Skip the values check if an integrator.integrate method is JIT
        # compiled with JAX
        if any(
            nam in type(integration_domain).__name__ for nam in ["Jaxpr", "JVPTracer"]
        ):
            return dim
        boundaries_are_invalid = (
            anp.min(integration_domain[:, 1] - integration_domain[:, 0]) < 0.0
        )
        # Skip the values check if an integrator.integrate method is
        # compiled with tensorflow.function
        if type(boundaries_are_invalid).__name__ == "Tensor":
            return dim
        if boundaries_are_invalid:
            raise ValueError("integration_domain has invalid boundary values")
        return dim


class RNG:
    """
    A random number generator helper class for multiple numerical backends

    Notes:
    * The seed argument may behave differently in different versions of a
      numerical backend and when using GPU instead of CPU
      * https://pytorch.org/docs/stable/notes/randomness.html
      * https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator
      * https://www.tensorflow.org/api_docs/python/tf/random/Generator
        Only the Philox RNG guarantees consistent behaviour in Tensorflow.
    * For torch, the RNG state is global, so if VEGAS integration uses this and
      the integrand itself generates random numbers and changes the seed,
      the calculated grid points may no longer be random.
      Torch allows to fork the RNG, but this may be slow.
    * Often uniform random numbers are generated in [0, 1) instead of [0, 1].
      * numpy: random() is in [0, 1) and uniform() in [0, 1]
      * JAX: uniform() is in [0, 1)
      * torch: rand() is in [0, 1)
      * tensorflow: uniform() is in [0, 1)
    """

    def __init__(self, backend, seed=None):
        """Initialize a RNG which can be seeded and is stateful if the backend supports it

        Args:
            backend (string): Numerical backend, e.g. "torch".
            seed (int or None): Random number generation seed. If set to None, the RNG is seeded randomly if possible. Defaults to None.

        Returns:
            An object whose "uniform" method generates uniform random numbers for the given backend
        """
        if backend == "numpy":
            import numpy as np

            self._rng = np.random.default_rng(seed)
            self.uniform = lambda size, dtype: self._rng.random(size=size, dtype=dtype)
        elif backend == "torch":
            import torch

            if seed is None:
                torch.random.seed()
            else:
                torch.random.manual_seed(seed)
            self.uniform = lambda size, dtype: torch.rand(size=size, dtype=dtype)
        elif backend == "jax":
            from jax.random import PRNGKey, split, uniform

            if seed is None:
                # Generate a random seed; copied from autoray:
                # https://github.com/jcmgray/autoray/blob/35677037863d7d0d25ff025998d9fda75dce3b44/autoray/autoray.py#L737
                from random import SystemRandom

                seed = SystemRandom().randint(-(2 ** 63), 2 ** 63 - 1)
            self._jax_key = PRNGKey(seed)

            def uniform_func(size, dtype):
                self._jax_key, subkey = split(self._jax_key)
                return uniform(subkey, shape=size, dtype=dtype)

            self.uniform = uniform_func
        elif backend == "tensorflow":
            import tensorflow as tf

            if seed is None:
                self._rng = tf.random.Generator.from_non_deterministic_state()
            else:
                self._rng = tf.random.Generator.from_seed(seed)
            self.uniform = lambda size, dtype: self._rng.uniform(
                shape=size, dtype=dtype
            )
        else:
            if seed is not None:
                anp.random.seed(seed, like=backend)
            self._backend = backend
            self.uniform = lambda size, dtype: anp.random.uniform(
                size=size, dtype=get_dtype_name(dtype), like=self._backend
            )

    def uniform(self, size, dtype):
        """Generate uniform random numbers in [0, 1) for the given numerical backend.
        This function is backend-specific; its definitions are in the constructor.

        Args:
            size (list): The shape of the generated numbers tensor
            dtype (backend dtype): The dtype for the numbers, e.g. torch.float32

        Returns:
            backend tensor: A tensor with random values for the given numerical backend
        """
        pass

    def jax_get_key(self):
        """
        Get the current PRNGKey.
        This function is needed for non-determinism when JIT-compiling with JAX.
        """
        return self._jax_key

    def jax_set_key(self, key):
        """
        Set the PRNGKey.
        This function is needed for non-determinism when JIT-compiling with JAX.
        """
        self._jax_key = key


# Register anp.repeat for torch
@partial(register_function, "torch", "repeat")
def _torch_repeat(a, repeats, axis=None):
    import torch

    # torch.repeat_interleave corresponds to np.repeat and should not be
    # confused with torch.Tensor.repeat.
    return torch.repeat_interleave(a, repeats, dim=axis)
