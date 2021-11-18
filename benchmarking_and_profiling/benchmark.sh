#!/bin/bash

for integrator in "Boole"; do
	echo "Deleting previous measurements if needed"
	rm tmp_results.csv

	for backend in "numpy" "torch" "jax" "tensorflow"; do
		run_measurement() {
			echo "Executing measure_runtimes.py for integrator $integrator, backend $backend, extra args: $*"
			python3 measure_runtimes.py "$backend" "$integrator" sin_prod 3 \
				tmp_results.csv --base=1.618 "$@"
		}
		run_measurement "$@"
		if [[ "$backend" != "numpy" ]]; then
			run_measurement --compile-parts "$@"
			run_measurement --compile "$@"
		fi
	done

	echo "Executing plot_runtimes.py to generate plots"
	python3 plot_runtimes.py tmp_results.csv
done
