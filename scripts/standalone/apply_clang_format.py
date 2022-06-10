"""
Rather than simply checking for clang-format, this file executes it
and changes files
"""

from pathlib import Path
from subprocess import Popen

file_names = []
for fname in Path('src').glob('**/*.h'):
    file_names.append(str(fname))
for fname in Path('src').glob('**/*.cpp'):
    file_names.append(str(fname))

for file_name in file_names:
    Popen("clang-format -i " + file_name, shell=True).wait()
