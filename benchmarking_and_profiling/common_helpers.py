import sys

sys.path.append("..")

from autoray import numpy as anp

from torchquad import set_precision, enable_cuda


def setup_tpu(backend):
    """Set up TPU on Google Colab"""
    if backend == "tensorflow":
        import tensorflow as tf

        print("Setting up TPU for Tensorflow")
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tf.device("TPU").__enter__()
        tpu_worker = tpu.cluster_spec().as_dict()["worker"]
        default_device = tf.constant(1.0).device
        print(
            f"Tensorflow {tf.__version__}, TPU {tpu_worker}, default device {default_device}"
        )
        if "TPU" not in str(default_device):
            raise RuntimeError("TPU setup failed")
    elif backend == "torch":
        # It seems impossible to use the TPU by default in PyTorch,
        # e.g. via a global dtype or device setting.
        # Here the exception happens and torchquad does not work with
        # PyTorch/XLA on TPU.
        print("Trying PyTorch/xla setup")
        import torch
        import torch_xla.core.xla_model as xm

        tpu_device = xm.xla_device(devkind="TPU")
        xla_dtype = anp.array([1.0], like="torch", device=tpu_device).dtype
        print(f"xla_dtype: {xla_dtype}")
        try:
            torch.set_default_tensor_type(xla_dtype)
        except TypeError as err:
            print(f"Cannot set a TPU dtype as default dtype: {err}")
        default_device = anp.array([1.0], like="torch").device
        print(f"Default PyTorch device: {default_device}")
        if "xla" not in str(default_device):
            raise RuntimeError("Cannot run on TPU.")
    elif backend == "jax":
        print("Setting up TPU for JAX")
        import jax.tools.colab_tpu

        jax.tools.colab_tpu.setup_tpu()
        import jax

        print(
            f"First device type: {jax.devices()[0].device_kind}, devices: {jax.devices()}"
        )
        if not any("TPU" in str(dev.device_kind) for dev in jax.devices()):
            raise RuntimeError("Cannot run on TPU.")


def setup_for_backend(backend, precision_dtype, use_tpu):
    """
    Backend-specific initialisations, e.g. of CUDA, and precision configuration
    """
    if backend == "tensorflow":
        from tensorflow.python.ops.numpy_ops import np_config

        np_config.enable_numpy_behavior()
        if use_tpu:
            setup_tpu(backend)
    elif backend == "torch":
        if not use_tpu:
            enable_cuda()
        else:
            setup_tpu(backend)
    elif backend == "jax" and use_tpu:
        setup_tpu(backend)
    if precision_dtype in ["float32", "float64"]:
        precision = {"float32": "float", "float64": "double"}[precision_dtype]
        set_precision(precision, backend=backend)
    else:
        print(f"Not setting global precision for {backend}")


def gaussian(a, b, c, x):
    """
    Gaussian function, as defined at
    https://en.wikipedia.org/wiki/Gaussian_function

    Args:
        a (float): vertical size
        b (backend tensor of shape (dim,)): centre position
        c (float): horizontal size
        x (backend tensor of shape (N, dim)): points to be evaluated
    """
    off = x - b
    length_sqr = anp.sum(off * off, axis=1)
    return a * anp.exp(-length_sqr / (2.0 * c**2))


def _get_gaussian_peaks_params():
    centres = [[0.0, 0.0], [0.5, 0.5], [0.6, 0.8], [1.0, 0.2]]
    vertical_sizes = [2.0, 0.3, 1.0, 1.0]
    horizontal_sizes = [0.1, 0.3, 0.05, 0.05]
    for k in range(21):
        cx = k / 20.0
        cy = cx
        centres.append([cx, cy])
        vertical_sizes.append(1.0 * (1.0 if k % 2 == 0 else -1.0))
        horizontal_sizes.append(0.05)
    return centres, vertical_sizes, horizontal_sizes


GAUSSIAN_PEAK_PARAMS = _get_gaussian_peaks_params()


def integrand_gaussian_peaks(x):
    """
    An example integrand function which is costly to evaluate.
    It also requires a big amount of memory.
    """
    centres, vertical_sizes, horizontal_sizes = GAUSSIAN_PEAK_PARAMS
    result = None
    for a, b, c in zip(vertical_sizes, centres, horizontal_sizes):
        y = gaussian(a, anp.array(b, like=x, dtype=x.dtype), c, x)
        if result is None:
            result = y
        else:
            result = result + y
    return result


def integrand_vegas_peak(x):
    """
    Example 4D integrand from
    https://vegas.readthedocs.io/en/latest/tutorial.html#basic-integrals
    x is changed so that the domain is in [0,1]^4 instead of [-1,1]×[0,1]^3
    """
    x = anp.concatenate([x[:, 0:1] * 2.0 - 1.0, x[:, 1:]], axis=1, like=x)
    dx2 = anp.sum((x - 0.5) ** 2, axis=1)
    return anp.exp(-dx2 * 100.0) * 1013.2118364296088 * 2.0


def integrand_sqs(x):
    """
    An example integrand function which is moderate to evaluate for high dimensions.
    """
    y = anp.sqrt(x[:, 0])
    for dim in range(1, x.shape[1]):
        y = anp.sqrt((2.3 * dim) * y + x[:, dim])
    return y


def integrand_sin_prod(x):
    """
    An example integrand function which is moderate to evaluate.
    """
    return anp.prod(anp.sin(x), axis=1)


def integrand_cos_prod(x):
    """
    Similar to sin_prod but simple reference solution if the domain starts at 0
    """
    return anp.prod(anp.cos(x), axis=1)


def integrand_sum(x):
    """
    An example integrand function which is cheap to evaluate.
    It may be too cheap which could lead to optimizing it away in e.g. JAX.
    """
    return anp.sum(x, axis=1)


def integrand_firstdim(x):
    """
    An example integrand function which is very cheap to evaluate but may cause
    optimizing away interesting part of the calculations
    """
    return x[:, 0]


integrand_functions = {
    "gaussian_peaks": integrand_gaussian_peaks,
    "vegas_peak": integrand_vegas_peak,
    "sqs": integrand_sqs,
    "sin_prod": integrand_sin_prod,
    "cos_prod": integrand_cos_prod,
    "sum": integrand_sum,
    "firstdim": integrand_firstdim,
    "zero": lambda x: x[:, 0] * 0.0,
    "one": lambda x: x[:, 0] * 0.0 + 1.0,
    "four": lambda x: x[:, 0] * 0.0 + 4.0,
}


def get_reference_integral(integrand_name, domain):
    """Calculate a reference solution for one of the example integrands"""
    backend = "numpy"
    domain = anp.array(domain, like=backend)
    if integrand_name == "cos_prod":
        if all(arr[0] == 0.0 for arr in domain):
            return anp.prod(anp.sin(domain[:, 1]))
        else:
            raise NotImplementedError(
                "no reference solution implemented for a domain which doesn't start at 0.0"
            )
    elif integrand_name == "gaussian_peaks":
        assert (
            anp.max(anp.abs(domain - anp.array([[0.0, 1.0], [0.0, 1.0]], like=domain)))
            == 0.0
        )
        # Solution calculated with Boole, numpy, float64, 2825^2 points
        return 0.19373664015937822
    elif integrand_name == "vegas_peak":
        assert domain.shape == (4, 2), "only 4D supported"
        assert (
            anp.max(anp.abs(domain - anp.array([[0.0, 1.0]] * 4, like=domain))) == 0.0
        )
        return 1.0
    elif integrand_name == "zero":
        return 0.0
    elif integrand_name == "one":
        return anp.prod(domain[:, 1] - domain[:, 0])
    elif integrand_name == "four":
        return anp.prod(domain[:, 1] - domain[:, 0]) * 4.0
    else:
        raise NotImplementedError(
            f"no reference solution implemented for {integrand_name}"
        )
