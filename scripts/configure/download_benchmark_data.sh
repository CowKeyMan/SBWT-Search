#!/bin/bash
mkdir -p benchmark_objects
cd benchmark_objects

GRCh38="GRCh38_latest_genomic.fna"
# Source: https://www.ncbi.nlm.nih.gov/projects/genome/guide/human/index.shtml
wget -nc https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/${GRCh38}.gz
if test -f "${GRCh38}"; then
        echo "${GRCh38} already unzipped, skipping "
else
        echo "Unzipping ${GRCh38}"
        gzip -dk "${GRCh38}.gz" > "${GRCh38}"
fi
cd ..
