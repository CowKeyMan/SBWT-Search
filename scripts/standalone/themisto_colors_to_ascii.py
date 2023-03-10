#!/bin/python3

"""
Converts (sorted) output from themisto into the same output format as this repo
"""

import argparse
from contextlib import ExitStack

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='The file to convert', required=True
)
parser.add_argument(
    '-o',
    '--output-prefix',
    help='Where to store the output. ".txt" will be appended to this',
    required=True
)
args = vars(parser.parse_args())

u64_bytes = 8
byteorder = 'little'

with ExitStack() as stack:
    in_file = stack.enter_context(
        open(args['input'], mode='r', encoding="utf-8")
    )
    out_file = stack.enter_context(
        open(args['output_prefix'] + '.txt', mode='wb')
    )
    mode_str = 'ascii'
    ver_str = 'v1.0'
    out_file.write(
        (len(mode_str)).to_bytes(u64_bytes, byteorder='little', signed=False)
    )
    out_file.write(mode_str.encode('ascii'))
    out_file.write(
        (len(ver_str)).to_bytes(u64_bytes, byteorder='little', signed=False)
    )
    out_file.write(ver_str.encode('ascii'))
    for i, line in enumerate(in_file):
        if i % 2 == 0:
            print(i, end=' ', flush=True)
        numbers = [int(x) for x in line.split(' ')]
        out_file.write(
            (' '.join([str(x) for x in (numbers[1:])]) + '\n')
            .encode('ascii')
        )
