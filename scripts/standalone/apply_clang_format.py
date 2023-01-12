#!/bin/python3

"""
Applies the changes proposed by clang format. Warning: This actually changes
the source code. While it would not change the contents, make sure that you
agree with the changes (you can check what changes will be applied if you run
./scripts/static_analysers/clang_format.py)
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
    with Popen("clang-format -i " + str(file_name), shell=True) as p:
        p.wait()
