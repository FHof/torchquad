#!/usr/bin/env bash

# Pyprof profiling with nsys as described at
# https://docs.nvidia.com/deeplearning/frameworks/pyprof-user-guide/profile.html
profile_with_nsys() {
	output_folder=$1
	shift
	mkdir -p "$output_folder"
	nsys profile --trace cuda,nvtx,osrt,cudnn,cublas --force-overwrite true \
		--capture-range cudaProfilerApi --capture-range-end stop \
		--export sqlite --output "${output_folder}/out_nsys" \
		./do_profile.py torch float32 Boole --use-pyprof \
		--output-folder "$output_folder" "$@"
	python3 -m pyprof.parse "${output_folder}/out_nsys.sqlite" \
		> "${output_folder}/out_nsys.dict"
}

profile_with_nsys "./out_nsys_compile_parts" --compile-parts
profile_with_nsys "./out_nsys_compile_all" --compile-all
profile_with_nsys "./out_nsys_uncompiled"
