#!/bin/python3

"""
Goes through the specified files in the repository, reads the first comment,
and generates a documentation page file tree with the directory structure and
the comment next to the file. This makes it easier to generate documentation
where the user can see all the file descriptions in one documentation page
"""

from fnmatch import fnmatchcase
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser

exit_code = 0

parser = ArgumentParser()
parser.add_argument('--no-print', default=False, action='store_true')
args = vars(parser.parse_args())

can_print = True
if args['no_print']:
    can_print = False


def post_process_comment(comment: list[str]) -> str:
    for i, c in enumerate(comment):
        if c.startswith('|'):
            comment[i] = '\n' + c
    return '\n'.join(
        [c.strip() for c in (" ".join(comment)).split('\n')]
    )


def get_cpp_comment_lines(filename: str) -> list[str]:
    with open(filename, 'r', encoding="utf-8") as f:
        lines = [line[:-1] for line in f.readlines()]
    comment = []
    assert lines[3] == "/**", "4th line should be /**"
    assert lines[4].startswith(" * @file "), \
        "5th line should start with ' * @file'"
    first_line_starting_preamble = " * @brief"
    assert lines[5].startswith(first_line_starting_preamble), \
        f"6th line should start with {first_line_starting_preamble}"
    comment.append(lines[5][len(first_line_starting_preamble):])
    for line in lines[6:]:
        if line == " */":
            break
        starting_preamble = " * "
        assert line.startswith(starting_preamble), \
            f"docstring lines should start with {starting_preamble}"
        comment.append(line[len(starting_preamble):])
    return comment


def get_python_comment_lines(filename: str) -> list[str]:
    with open(filename, 'r', encoding="utf-8") as f:
        lines = [line[:-1] for line in f.readlines()]
    comment = []
    assert lines[0] == "#!/bin/python3", "First line should be #!/bin/python3"
    assert lines[1] == "", "2nd line should be blank"
    assert (
        (lines[2] == '"""', "3rd line should be start of docstring")
        or (lines[2] == 'r"""', "3rd line should be start of docstring")
    )
    for line in lines[3:]:
        if line == '"""':
            break
        comment.append(line)
    return comment


def get_bash_comment_lines(filename: str) -> list[str]:
    with open(filename, 'r', encoding="utf-8") as f:
        lines = [line[:-1] for line in f.readlines()]
    comment = []
    assert lines[0] == "#!/bin/bash", "First line should be #!/bin/bash"
    assert lines[1] == "", "2nd line should be blank"
    for line in lines[2:]:
        if not line.startswith("#"):
            break
        assert line.startswith("# "), "docstring lines should start with #"
        comment.append(line[2:])
    return comment


def get_hash_commented_comment_lines(filename: str) -> list[str]:
    with open(filename, 'r', encoding="utf-8") as f:
        lines = [line[:-1] for line in f.readlines()]
    comment = []
    for line in lines:
        if not line.startswith("#"):
            break
        assert line.startswith("# "), "docstring lines should start with #"
        comment.append(line[2:])
    return comment


def get_bat_comment_lines(filename: str) -> list[str]:
    with open(filename, 'r', encoding="utf-8") as f:
        lines = [line[:-1] for line in f.readlines()]
    comment = []
    for line in lines:
        if not line.startswith("Rem"):
            break
        assert line.startswith("Rem "), "docstring lines should start with Rem"
        comment.append(line[4:])
    return comment


def get_all_lines(filename: str) -> list[str]:
    with open(filename, 'r', encoding="utf-8") as f:
        return [line[:-1] for line in f.readlines()]


def get_git_tracked_files():
    proc = subprocess.run(
        "git branch --show-current".split(),
        capture_output=True,
        check=True
    )
    git_branch = proc.stdout.decode()
    proc = subprocess.run(
        f"git ls-tree -r {git_branch} --name-only".split(),
        capture_output=True,
        check=True
    )
    return proc.stdout.decode().split('\n')


file_regex_and_comment_function = [
    (["*.hpp", "*.cuh", "*.h"], get_cpp_comment_lines),
    (["*.sh", "*.sbatch", "pre-commit"], get_bash_comment_lines),
    (["*.py", "*.py.in"], get_python_comment_lines),
    (
        [
            "CMakeLists.txt",
            "*.cmake",
            ".clang*",
            ".gitignore",
            "*.yml",
            "Doxyfile.in",
        ],
        get_hash_commented_comment_lines
    ),
    ([".dir_docstring"], get_all_lines),
    (["*.bat"], get_bat_comment_lines),
]


def extract_comment(filename: str, full_filename: str) -> str:
    for patterns, comment_function in file_regex_and_comment_function:
        for pattern in patterns:
            if fnmatchcase(filename, pattern):
                return post_process_comment(comment_function(full_filename))
    return None


class Contents:
    def __init__(self):
        self.directories: dict[str, Contents] = defaultdict(Contents)
        self.filename_and_comment: list[tuple[str, str]] = []
        self.file_count = 0
        self.description = ""

    def add_file(self, filename: str, full_filename: str = None) -> int:
        if full_filename is None:
            full_filename = filename
        slash_index = filename.find('/')
        if slash_index == -1:
            comment = extract_comment(filename, full_filename)
            if filename == ".dir_docstring":
                if comment is not None:
                    self.description = comment
                self.file_count += 1
            elif comment is not None:
                self.filename_and_comment.append((filename, comment))
                self.file_count += 1
        else:
            folder = filename[:slash_index]
            filename_truncated = filename[slash_index + 1:]
            count = self.directories[folder].add_file(
                filename_truncated, full_filename
            )
            if count > 0:
                self.file_count += 1
        return self.file_count

    def print(self, tabs: int = 0) -> None:
        tab = '  '
        if not can_print:
            return
        print()
        for filename, comment in self.filename_and_comment:
            comment = comment.replace('\n', '\n' + tab)
            for _ in range(tabs):
                print(tab, end='')
                comment = comment.replace('\n', '\n' + tab)
            comment = comment.replace('|', '', 1)
            newline = "\n"
            additional = "| " if newline in comment else ''
            print(f'* {additional}`{filename}`: {comment}')
        for dir_name, contents in sorted(
            self.directories.items(), key=lambda x: x[0]
        ):
            if contents.file_count == 0:
                continue
            for _ in range(tabs):
                print(tab, end='')
            if len(contents.description) > 0:
                print(f'* **{dir_name}**: {contents.description}')
            else:
                print(f'* **{dir_name}**:')
            contents.print(tabs + 1)
            if len(contents.directories) == 0:
                print()


if __name__ == '__main__':
    try:
        all_files = get_git_tracked_files()
    except subprocess.CalledProcessError:
        print("git command not found!", file=sys.stderr)
        exit_code = 1

    # Populate contents
    root = Contents()
    for filename in all_files:
        if not Path(filename).exists():
            continue
        try:
            root.add_file(filename)
        except AssertionError as e:
            print(f'Error in file: {filename}', e.args[0], file=sys.stderr)
            exit_code = 1

    print()
    root.print()

    sys.exit(exit_code)
