"""This file contains various utility functions for the integrations methods."""

from autoray import numpy as anp
from autoray import infer_backend
from loguru import logger


def _linspace_with_grads(start, stop, N, requires_grad):
    """Creates an equally spaced 1D grid while keeping gradients
    in regard to inputs.
    Args:
        start (backend tensor): Start point (inclusive).
        stop (backend tensor): End point (inclusive).
        N (backend tensor): Number of points.
        requires_grad (bool): Indicates if output should be recorded for backpropagation.
    Returns:
        backend tensor: Equally spaced 1D grid
    """
    if requires_grad:
        # Create 0 to 1 spaced grid
        grid = anp.linspace(0, 1, N, like=start)

        # Scale to desired range, thus keeping gradients
        grid *= stop - start
        grid += start

        return grid
    else:
        return anp.linspace(start, stop, N, like=start)


def _setup_integration_domain(dim, integration_domain, backend):
    """Sets up the integration domain if unspecified by the user.
    Args:
        dim (int): Dimensionality of the integration domain.
        integration_domain (list or backend tensor, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim. It also determines the numerical backend if possible.
        backend (string): Numerical backend. This argument is ignored if the backend can be inferred from integration_domain.
    Returns:
        backend tensor: Integration domain.
    """

    # Store integration_domain
    # If not specified, create [-1,1]^d bounds
    logger.debug("Setting up integration domain.")
    if integration_domain is not None:
        if len(integration_domain) != dim:
            raise ValueError(
                "Dimension and length of integration domain don't match. Should be e.g. dim=1 dom=[[-1,1]]."
            )
        if infer_backend(integration_domain) == "builtins":
            return anp.array(integration_domain, like=backend)
        return integration_domain
    else:
        return anp.array([[-1, 1]] * dim, like=backend)


class _RNG:
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

    def __init__(self, seed, backend):
        """Initialize a RNG which can be seeded and is stateful if the backend supports it

        Args:
            seed (int or None): Random number generation seed. If set to None, the RNG is seeded randomly if possible. Defaults to None.
            backend (string): Numerical backend, e.g. "torch".

        Returns:
            An object whose "uniform" method generates uniform random numbers for the given backend
        """
        if backend == "numpy":
            import numpy as np

            self._rng = np.random.default_rng(seed)
            self.uniform = lambda size: self._rng.uniform(size=size)
        elif backend == "torch":
            import torch

            if seed is None:
                torch.random.seed()
            else:
                torch.random.manual_seed(seed)
            self.uniform = lambda size: torch.rand(size=size)
        elif backend == "jax":
            from jax.random import PRNGKey, split, uniform

            if seed is None:
                # Generate a random seed; copied from autoray:
                # https://github.com/jcmgray/autoray/blob/35677037863d7d0d25ff025998d9fda75dce3b44/autoray/autoray.py#L737
                from random import SystemRandom

                seed = SystemRandom().randint(-(2 ** 63), 2 ** 63 - 1)
            self._jax_key = PRNGKey(seed)

            def uniform_func(size):
                self._jax_key, subkey = split(self._jax_key)
                return uniform(subkey, shape=size)

            self.uniform = uniform_func
        elif backend == "tensorflow":
            import tensorflow as tf

            if seed is None:
                self._rng = tf.random.Generator.from_non_deterministic_state()
            else:
                self._rng = tf.random.Generator.from_seed(seed)
            self.uniform = lambda size: self._rng.uniform(shape=size)
        else:
            if seed is not None:
                anp.random.seed(seed, like=backend)
            self._backend = backend
            self.uniform = lambda size: anp.random.uniform(
                size=size, like=self._backend
            )
