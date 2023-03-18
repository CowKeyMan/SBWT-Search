#!/bin/bash

# Run the benchmark_main several times and save the results to a file. Called by
# scipts/sbatch/benchmark.sbatch It is expected that the *.tdbg file is within
# benchmark_objects/index folder

if [ $# -ne 1 ]; then
  echo "Please enter a single argument to this script, which is the output file location"
  exit 1
fi

## Set loglevel to trace because we use this to get timing statistics
export SPDLOG_LEVEL=TRACE

benchmark_out="$1"

if [ ! -d "benchmark_objects/index" ] || [ ! -f "benchmark_objects/index/index.tdbg" ]; then
  echo "This script expects a 'benchmark_objects/index' folder with 'index.tdbg' to be present within that folder"
  exit 1
fi

colors_file="benchmark_objects/index/index_d1.tcolors"
input_files=(
  "benchmark_objects/list_files/input/index_search_results_d1.list"
)
output_file="benchmark_objects/list_files/output/color_search_results.list"
printing_modes=(
  "ascii"
  # "binary"
)
devices=(
  "nvidia"
  "cpu"
  # "amd"
)

streams_options=(1 2 4 6 8 10 12 14 16)

for device in "${devices[@]}"; do
  ./scripts/build/release_${device}.sh >&2
  for streams in "${streams_options[@]}"; do
    for input_file in "${input_files[@]}"; do
      for printing_mode in "${printing_modes[@]}"; do
        echo "Now running: File ${input_file} with ${streams} streams in ${printing_mode} format on ${device} device"
        echo "Now running: File ${input_file} with ${streams} streams in ${printing_mode} format on ${device} device" >> "${benchmark_out}"
        ./build/bin/sbwt_search colors \
          -i "${colors_file}" \
          -q "${input_files}" \
          -o "${output_file}" \
          -s "${streams}" \
          -p "${printing_mode}" \
          >> "${benchmark_out}"
        printf "Size of outputs: "
        ls -lh "benchmark_objects/running" | head -1
        if [ "${printing_mode}" = "ascii" ]; then
          sed -i 's/-2/-1/g' benchmarl_objects/running/*
          diff -qr "benchmark_objects/running" "benchmark_objects/color_search_results_t0.7"
        fi
        rm benchmark_objects/running/*
      done
    done
  done
done