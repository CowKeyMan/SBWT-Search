#!/bin/python3

"""
Converts output from algbio/sbwt to the ascii format produced by this program
"""

import argparse
from contextlib import ExitStack

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='The file to convert', required=True
)
parser.add_argument(
    '-o', '--output-prefix', help='Where to store the output.', required=True
)
args = vars(parser.parse_args())

u64_bytes = 8
byteorder = 'little'

with ExitStack() as stack:
    in_file = stack.enter_context(
        open(args['input'], mode='r', encoding="utf-8")
    )
    out_file = stack.enter_context(
        open(args['output_prefix'], mode='wb')
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
        line = line.strip() + '\n'
        out_file.write(line.encode('ascii'))
