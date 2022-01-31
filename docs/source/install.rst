.. _installation:

Getting started
===============

This is a brief introduction on how to set up *torchquad*.

Prerequisites
--------------

*torchquad* is built with

- `autoray <https://github.com/jcmgray/autoray>`_, which means the implemented quadrature supports `Numpy <https://numpy.org/>`_ and can be used for machine learning with modules such as `PyTorch <https://pytorch.org/>`_, `JAX <https://github.com/google/jax/>`_ and `Tensorflow <https://www.tensorflow.org/>`_, where it is fully differentiable
- `conda <https://docs.conda.io/en/latest/>`_, which will take care of all requirements for you

We recommend using `conda <https://docs.conda.io/en/latest/>`_, especially if you want to utilize the GPU.
It will automatically set up CUDA and the cudatoolkit for you in that case.
Note that *torchquad* also works on the CPU; however, it is optimized for GPU usage.
Currently torchquad only supports NVIDIA cards with CUDA. We are investigating future support for AMD cards through `ROCm <https://pytorch.org/blog/pytorch-for-amd-rocm-platform-now-available-as-python-package/>`_.

For a detailed list of required packages, please refer to the `conda environment file <https://github.com/esa/torchquad/blob/main/environment.yml>`_.

Installation
-------------

First, we must make sure we have `torchquad <https://github.com/esa/torchquad>`_ installed.
The easiest way to do this is simply to

   .. code-block:: bash

      conda install torchquad -c conda-forge -c pytorch

Note that since PyTorch is not yet on *conda-forge* for Windows, we have explicitly included it here using ``-c pytorch``.

Alternatively, it is also possible to use

   .. code-block:: bash

      pip install torchquad

NB Note that *pip* will **not** set up PyTorch with CUDA and GPU support. Therefore, we recommend to use *conda*.

**GPU Utilization**

With *conda* you can install the GPU version of PyTorch with ``conda install pytorch cudatoolkit -c pytorch``.
For alternative installation procedures please refer to the `PyTorch Documentation <https://pytorch.org/get-started/locally/>`_.

Usage
-----

Now you are ready to use *torchquad*.
A brief example of how *torchquad* can be used to compute a simple integral can be found on our `GitHub <https://github.com/esa/torchquad#usage>`_.
For a more thorough introduction, please refer to the `tutorial <https://torchquad.readthedocs.io/en/main/tutorial.html>`_.
