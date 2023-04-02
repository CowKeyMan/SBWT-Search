#!/bin/bash

# This script is used to test benchmarking locally. We download the pre made
# benchmark files from dropbox and unzip them locally

cd benchmark_objects

wget https://www.dropbox.com/sh/2007erh8vgqr5qz/AABrqNwZMP8HgM7noNWUmGkqa?dl=1 -O benchmark_objects.zip

unzip benchmark_objects.zip

rm benchmark_objects.zip
