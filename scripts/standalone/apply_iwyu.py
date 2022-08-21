"""
This file checks for incorrect includes in a file, folder or the entire repository and applies includ-what-you-use as well as clang-format to them
"""

from pathlib import Path
from subprocess import Popen
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', help='build folder where compilation_database is', required=True)
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


def generate_iwyu_for_item(item: str):
    Popen(
        f"iwyu_tool.py -p {args['p']} -j {args['j']} {item} > {iwyu_output}",
        shell=True
    ).wait()


def format_file(filename: str):
    if Path(filename).suffix[1:] in header_suffixes:
        generate_iwyu_for_item(str(Path(filename).parent))
    else:
        generate_iwyu_for_item(filename)
    Popen(f"fix_includes.py --nocomments --nosafe_headers {filename} < {iwyu_output} > /dev/null", shell=True).wait()
    Popen("clang-format -i " + str(filename), shell=True).wait()


def format_directory(directory: str):
    generate_iwyu_for_item(directory)
    filenames = get_files_in_dir(directory)
    filenames_string = ' '.join([str(f) for f in filenames])
    Popen(
        f"fix_includes.py --nocomments --nosafe_headers {filenames_string} < {iwyu_output} > /dev/null",
        shell=True
    ).wait()
    for filename in filenames:
        Popen("clang-format -i " + str(filename), shell=True).wait()


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


