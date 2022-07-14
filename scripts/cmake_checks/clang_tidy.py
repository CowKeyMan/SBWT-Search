"""
clang-tidy needs the files input manually. However, we do not want to put test
files in there, as these trigger a chain reaction to add the googltest imported
header files as well. Hence we simply glob for all the other .h and .cpp files
and put them through the tool

This script is executed automatically by cmake when building
"""

from pathlib import Path
from subprocess import Popen

file_names = ""
folders = set()
for fname in Path('src').rglob('*.h'):
    file_names += str(fname) + " "
    folders.add(str(fname.parent))
for fname in Path('src').rglob('*.hpp'):
    file_names += str(fname) + " "
for fname in Path('src').rglob('*.cuh'):
    file_names += str(fname) + " "
for fname in Path('src').rglob('[!(test)]*[!(test)].cpp'):
    file_names += str(fname) + " "
for fname in Path('src').rglob('[!(test)]*[!(test)].cu'):
    file_names += str(fname) + " "

# folders_string = ""
# for folder in folders:
    # folders_string += " -I " + folder


command = (
    "clang-tidy "
    + "-p ./build "
    + file_names
)
print('Running: ' + command, flush=True)
Popen(command, shell=True).wait()
