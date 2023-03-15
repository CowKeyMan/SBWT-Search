#!/bin/bash

# Download some test objects which should probably not be as part of the repository.
# This script makes sure to not download again if the target folder already exists.
# However, if any parameter at all is passed, this is ignored and the existing folder
# is deleted and replaced.

# download the themisto example
if [[ -d "test_objects/themisto_example" && $# > 0 ]]; then
  rm -rf "test_objects/themisto_example"
fi
if [[ -d "test_objects/themisto_example" ]]; then
  echo "Themisto example already downloaded"
else
  mkdir -p test_objects/themisto_example
  cd test_objects/themisto_example
  wget https://www.dropbox.com/sh/7p4zoc0zttktryz/AABzLfSH3czl3VS0I0nCcVJIa?dl=0 -O themisto_example.zip
  unzip themisto_example.zip
  rm themisto_example.zip
  cd -
fi

cd benchmark_objects
wget https://www.dropbox.com/sh/o3vxi0sqw68mxb3/AADkKY-xWjwy9kvbh6sI-lD1a?dl=0 -O benchmark_objects.zip
unzip benchmark_objects.zip
rm benchmark_objects.zip
cd ..
