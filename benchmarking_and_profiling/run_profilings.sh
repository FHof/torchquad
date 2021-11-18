#!/usr/bin/env bash

# With MonteCarlo and JAX the XLA_PYTHON_CLIENT_PREALLOCATE=false environment
# variable can avoid out of memory crashes

integrator="Boole"
for backend in torch jax tensorflow; do
	./do_profile.py "$backend" float32 "$integrator" --output-folder "profiling_output/uncompiled" "$@"
	./do_profile.py "$backend" float32 "$integrator" --compile-parts --output-folder "profiling_output/compile_parts" "$@"
	./do_profile.py "$backend" float32 "$integrator" --compile-all --output-folder "profiling_output/compile_all" "$@"
done
