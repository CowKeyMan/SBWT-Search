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
from subprocess import Popen, PIPE

suffixes = 'h cpp cu cuh hpp'.split()
file_names = [
    x
    for x in Path('src').rglob('*')
    if x.suffix[1:] in suffixes and 'main' not in str(x)
]

Path('tmp').mkdir(exist_ok=True)

for file_name in file_names:
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
    process = Popen(command, shell=True, stdout=PIPE)
    process.wait()
    Path(original_file).unlink()
    Path(tidy_file).unlink()
    output = process.communicate()[0].decode()
    if len(output) > 0:
        printed = f'Showing diff of: {file_name}'
        print(printed)
        print('=' * len(printed), flush=True)
        print(output)
        print()
