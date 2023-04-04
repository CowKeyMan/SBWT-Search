#!/bin/python3

r"""
Converts \*.tdbg files to \*.sbwt files by adding 'plain-matrix' at the
beginning
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='The file to convert', required=True
)
parser.add_argument(
    '-o', '--output', help='Where to store the output.', required=True
)
args = vars(parser.parse_args())

u64_bytes = 8
byteorder = 'little'

start_string = 'plain-matrix'

with open(args['output'], 'wb') as f:
    f.write(
        len(start_string).to_bytes(u64_bytes, byteorder='little', signed=False)
    )
    f.write(start_string.encode('ascii'))


os.system(f"cat {args['input']} >> {args['output']}")
