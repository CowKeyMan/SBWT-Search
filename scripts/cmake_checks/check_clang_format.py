"""
clang-format is a bit difficult to work with on its own. Its output is not
very easy to read. Hence, here we have created a script to make it easier
to work with.
It creates temporary files so that we see the difference between each file
individually, and then we format them as we see fit in stdout, to be easier
for the eyes and to see which file is being analyzed at the time

This script is executed automatically by cmake when building
"""

from pathlib import Path
from subprocess import Popen

file_names = []
for fname in Path('src').glob('**/*.h'):
    file_names.append(fname)
for fname in Path('src').glob('**/*.cpp'):
    file_names.append(fname)

Path('tmp').mkdir(exist_ok=True)

for file_name in file_names:
    printed = 'Showing diff of: ' + str(file_name)
    print(printed)
    print('=' * len(printed), flush=True)
    with open(file_name, 'r') as f:
        lines = [x.replace('\r\n', '\n') for x in f.readlines()]
    original_file = f'tmp/{file_name.stem}_original{file_name.suffix}'
    tidy_file = f'tmp/{file_name.stem}_tidy{file_name.suffix}'
    with open(original_file, 'w') as f:
        f.writelines(lines)
    # no pager: print all to stdout rather than interactive mode
    # no-index: use git diff on untracked files
    command = (
        f"clang-format {original_file} > {tidy_file};"
        "git --no-pager diff --no-index"
        f" --ignore-cr-at-eol -U2 {original_file} {tidy_file}"
    )
    Popen(command, shell=True).wait()
    print()
    Path(original_file).unlink()
    Path(tidy_file).unlink()
