#!/bin/python3

"""
| Checks that the header files start with:
|
| #ifndef FILE_NAME_CAPITALIZED_<EXTENSION>
| #define FILE_NAME_CAPITALIZED_<EXTENSION>
|
| /**
|  * @file FileName.<extension>
|  * @brief <description goes here and on next line>
|
| and ends with:
|
| #endif
|
| The term FILE_NAME_CAPITALIZED_<EXTENSION> would be replaced by the filename.
So if we have a file called MyClass.h, it would be MY_CLASS_H
The steps are:
|     * Remove the .h
|     * Convert from camel case to snake case
|     * capitalise everything
|     * add a _H at the end
|
| If the file is a cuh file, it is the same as above
but it will have _CUH at the end
|
| The script also checks that python files start with:
|
| #!/usr/bin/python3
| <new line here>
| <3 " here, ie a docstring>
|
| And bash files should start with the following:
|
| #!/usr/bin/bash
| <new line here>
| # some comment
"""

import sys
from pathlib import Path


cpp_headers = (
    list(Path('src').glob('**/*.h'))
    + list(Path('src').glob('**/*.hpp'))
    + list(Path('src').glob('**/*.cuh'))
)


def capitalise(name: str, suffix: str) -> str:
    result = name[0]
    for char in name[1:]:
        if char.isupper() or not char.isalpha() and char != '_':
            result += '_'
        result += char.upper()
    result += '_' + suffix.upper()
    return result


exit_code = 0


def print_if_first_time():
    if exit_code == 0:
        print(
            '\n'
            '####################### ERROR ########################\n'
            'Some header files are misformatted! Please fix these!\n'
            '######################################################',
        )


for file_name in cpp_headers:
    with open(file_name, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    header_capitalized = capitalise(file_name.stem, file_name.suffix[1:])
    if len(lines) < 0 or not (
        lines[0].strip() == '#ifndef ' + header_capitalized
        and lines[1].strip() == '#define ' + header_capitalized
        and lines[-1].strip() == '#endif'
    ):
        print_if_first_time()
        exit_code = 1
        print(
            '* ' + str(file_name) + ' does not have proper header guards',
        )
    if len(lines) > 3 and not (
        lines[3].strip() == '/**'
        and lines[4].strip() == f'* @file {file_name.name}'
        and lines[5].startswith(' * @brief')
    ):
        print_if_first_time()
        exit_code = 1
        print(
            '* ' + str(file_name) + ' does not have proper docstring',
        )

python_files = (
    list(Path('src').glob('**/*.py'))
    + list(Path('src').glob('**/*.py.in'))
    + list(Path('scripts').glob('**/*.py'))
    + list(Path('scripts').glob('**/*.py.in'))
)

for file_name in python_files:
    with open(file_name, 'r', encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    if len(lines) < 3 or not (
        lines[0] == "#!/bin/python3"
        and lines[1] == ""
        and lines[2] == '"""' or lines[2] == 'r"""'
    ):
        print_if_first_time()
        exit_code = 1
        print('* ' + str(file_name) + ' does not have proper headers')

bash_files = (
    list(Path('src').glob('**/*.sh'))
    + list(Path('scripts').glob('**/*.sh'))
)

for file_name in bash_files:
    with open(file_name, 'r', encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    if len(lines) < 3 or not (
        lines[0] == "#!/bin/bash"
        and lines[1] == ""
        and lines[2].startswith('# ')
    ):
        print_if_first_time()
        exit_code = 1
        print('* ' + str(file_name) + ' does not have proper headers')

if exit_code == 0:
    print('All headers OK!\n')

sys.exit(exit_code)
