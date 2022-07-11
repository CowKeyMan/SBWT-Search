"""
Rather than simply checking for clang-format, this file executes it
and changes files
"""

from pathlib import Path
from subprocess import Popen

suffixes = 'h cpp cu cuh hpp'.split()
file_names = [
    x
    for x in Path('src').rglob('*')
    if x.suffix[1:] in suffixes and 'main' not in str(x)
]


for file_name in file_names:
    Popen("clang-format -i " + str(file_name), shell=True).wait()
