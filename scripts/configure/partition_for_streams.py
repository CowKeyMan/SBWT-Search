#!/bin/python3

"""
A script to take a partitioned list file (includes single partition) and
divides it into the target partitions, the target partition is given as an
input to this script. Input files have a corresponding output file, and these
are also sorted accordingly. Note that the partitioning is done inplace to the
given files. A partitioned list file contains a path to a file in each line,
where a line with just a '/' symbol indicates the start of a new partitioning.
This partitioning is useful when we are using streams within our program.
"""

from pathlib import Path
import argparse
import sys
from collections import defaultdict
from contextlib import ExitStack
import prtpy

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input',
    help='Input list file',
    type=str,
    required=True
)
parser.add_argument(
    '-o', '--output',
    help='Output file corresponding to the list file',
    type=str,
    required=True
)
parser.add_argument(
    '-p', '--partitions',
    help='Number of partitions, must be at least 1',
    type=int,
    required=True
)
args = vars(parser.parse_args())


def get_all_files(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    result = []
    for line in lines:
        line = line.strip()
        if len(line) > 0 and line == '/':
            continue
        result.append(line)
    return result


input_files = get_all_files(args['input'])
output_files = get_all_files(args['output'])

if len(input_files) != len(output_files):
    print("Number of input and output files differ")
    sys.exit(1)

size_to_files = defaultdict(list)
for in_file, out_file in zip(input_files, output_files):
    size_to_files[Path(in_file).stat().st_size].append((in_file, out_file))

bins = prtpy.partition(
    algorithm=prtpy.partitioning.greedy,
    numbins=args['partitions'],
    items=size_to_files.keys()
)

with ExitStack() as stack:
    f_in = stack.enter_context(
        open(args['input'], mode='w', encoding='utf-8')
    )
    f_out = stack.enter_context(
        open(args['output'], mode='w', encoding='utf-8')
    )
    for i, b in enumerate(bins):
        if i > 0:
            f_in.write('/\n')
            f_out.write('/\n')
        for s in b:
            in_file, out_file = size_to_files[s].pop()
            f_in.write(in_file + '\n')
            f_out.write(out_file + '\n')
