"""
Rather than simply checking for clang-format, this file executes it
and changes files
"""

from pathlib import Path
from subprocess import Popen

suffixes = 'h cpp cu cuh'.split()
file_names = [
    str(x)
    for x in Path('src').rglob('*')
    if x.suffix[1:] in suffixes
]

for file_name in file_names:
    Popen("clang-format -i " + file_name, shell=True).wait()
