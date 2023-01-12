#!/bin/python3

"""
This file checks for incorrect includes in a file,
folder or the entire repository and applies include-what-you-use
as well as clang-format to them

THIS IS NOT RECOMMENDED TO USE YET (at least not until iwyu gets out of alpha
and is ready for production)
"""

from pathlib import Path
from subprocess import Popen
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p', help='build folder where compilation_database is', required=True
)
parser.add_argument('-j', help='jobs', required=False, type=int, default=8)
parser.add_argument('filenames', nargs=argparse.REMAINDER)
args = vars(parser.parse_args())

header_suffixes = 'h cuh hpp'.split()
implementation_suffixes = 'cpp cu'.split()
suffixes = header_suffixes + implementation_suffixes

iwyu_output = '/tmp/iwyu_out.txt'


def get_files_in_dir(directory: str):
    return [
        x
        for x in Path(directory).rglob('*')
        if x.suffix[1:] in suffixes
    ]


def generate_iwyu(filename: str):
    with Popen(
        f"iwyu_tool.py -p {args['p']} -j "
        f"{args['j']} {filename} > {iwyu_output}",
        shell=True
    ) as p:
        p.wait()


def format_file(filename: str):
    if Path(filename).suffix[1:] in header_suffixes:
        generate_iwyu(str(Path(filename).parent))
    else:
        generate_iwyu(filename)
    with Popen(
        "fix_includes.py --nocomments --nosafe_headers "
        f"{filename} < {iwyu_output} > /dev/null",
        shell=True
    ) as p:
        p.wait()
    with Popen("clang-format -i " + str(filename), shell=True) as p:
        p.wait()


def format_directory(directory: str):
    generate_iwyu(directory)
    filenames = get_files_in_dir(directory)
    filenames_string = ' '.join([str(f) for f in filenames])
    with Popen(
        "fix_includes.py --nocomments --nosafe_headers "
        f"{filenames_string} < {iwyu_output} > /dev/null",
        shell=True
    ) as p:
        p.wait()
    for filename in filenames:
        with Popen("clang-format -i " + str(filename), shell=True) as p:
            p.wait()


if __name__ == '__main__':
    if len(args['filenames']) == 0:
        format_directory('src')
    for item in args['filenames']:
        if Path(item).is_dir():
            format_directory(item)
        elif Path(item).is_file():
            format_file(item)
        else:
            print(f'Argument is invalid. "{item}" not found', file=sys.stderr)
