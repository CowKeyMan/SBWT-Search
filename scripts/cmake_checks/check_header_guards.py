"""
Checks that the header files start with:

#ifndef FILE_NAME_CAPITALIZED_H
#define FILE_NAME_CAPITALIZED_H

and ends with:

#endif

The term FILE_NAME_CAPITALIZED_H would be replaced by the filename.
So if we have a file called MyClass.h, it would be MY_CLASS_H
The steps are:
    * Remove the .h
    * Convert from camel case to snake case
    * capitalize everything
    * add a _H at the end

If the file is a cuh file, it is the same as above
but it will have _CUH at the end

This script is executed automatically by cmake when building
"""


from pathlib import Path

file_names = (
    list(Path('src').glob('**/*.h'))
    + list(Path('src').glob('**/*.hpp'))
    + list(Path('src').glob('**/*.cuh'))
)


def capitalize(name: str, suffix: str) -> str:
    result = name[0]
    for char in name[1:]:
        if char.isupper() or not char.isalpha():
            result += '_'
        result += char.upper()
    result += '_' + suffix.upper()
    return result


first_time = True

for file_name in file_names:
    with open(file_name, 'r') as f:
        lines = f.readlines()
    header_capitalized = capitalize(file_name.stem, file_name.suffix[1:])
    if not(
        lines[0].strip() == '#ifndef ' + header_capitalized
        and lines[1].strip() == '#define ' + header_capitalized
        and lines[-1].strip() == '#endif'
    ):
        if first_time:
            first_time = False
            print(
                '\n'
                '####################### ERROR ########################\n'
                'Some files are missing header guards. Please include these!\n'
                '######################################################',
            )
        print(
            '* ' + str(file_name) + ' does not have proper header guards',
        )

if first_time:
    print('All header guards OK!\n')
