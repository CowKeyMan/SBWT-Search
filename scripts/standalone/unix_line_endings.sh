#!/bin/bash

# run twice as sometimes once is not enough
# Only run on tracked files

dos2unix `git ls-tree -r master --name-only`
dos2unix `git ls-tree -r master --name-only`
