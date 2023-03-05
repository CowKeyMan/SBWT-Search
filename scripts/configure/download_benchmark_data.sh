#!/bin/bash

# Download the data used for benchmarking from public databases. The dataset
# file list is contained `scripts/configure/benchmark_datra_list.txt`.  This
# file contains all the files obtained from
# https://www.ebi.ac.uk/ena/browser/view/PRJEB32631.  From the paper: Mäklin,
# T., Thorpe, H. A., Pöntinen, A. K., Gladstone, R. A., Shao, Y., Pesonen, M.,
# … Corander, J. (2022). Strong pathogen competition in neonatal gut
# colonisation. Nature Communications, 13(1), 7417.
# doi:10.1038/s41467-022-35178-5.  The original dataset is 2TB big, so I have
# chosen to only have the first 20 files


mkdir -p benchmark_objects
cd benchmark_objects

wget -nc -i scripts/configure/benchmark_data_list.txt

cd ..
