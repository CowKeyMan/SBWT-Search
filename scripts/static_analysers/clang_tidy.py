#!/bin/python3

"""
clang-tidy needs the files input manually. However, we do not want to put test
files in there, as these trigger a chain reaction to add the googltest imported
header files as well. Hence we simply glob for all the other .h and .cpp files
and put them through the tool

This script is executed automatically by cmake when building
"""

from pathlib import Path
from subprocess import Popen, PIPE
import sys

file_names = ""
folders = set()
for fname in Path('src').rglob('*.h'):
    file_names += str(fname) + " "
    folders.add(str(fname.parent))
for fname in Path('src').rglob('*.hpp'):
    file_names += str(fname) + " "
for fname in Path('src').rglob('*.cuh'):
    file_names += str(fname) + " "
for fname in Path('src').rglob('*.cpp'):
    file_names += str(fname) + " "
for fname in Path('src').rglob('*.cu'):
    file_names += str(fname) + " "

command = (
    "clang-tidy "
    "-p ./build "
    + file_names
)
print('Running: ' + command, flush=True)
with Popen(command, shell=True, stdout=PIPE, stderr=PIPE) as p:
    out, _ = p.communicate()

# postprocess output
out = out.decode("utf-8")
print(out)

if out == '':
    print("clang-tidy passed")

sys.exit(out != '')
