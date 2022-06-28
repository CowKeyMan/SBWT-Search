#!/bin/bash

MODE=1 # 1 = the proper mode
# If any argument at all is passed to this script, it will instead run the 'debug' version, ie it will only download some small files, for local testing
if [ $# -ne 0 ]; then
	echo $#
	MODE=0
fi

mkdir -p benchmark_objects
cd benchmark_objects

# Download Fasta
FASTA1GB="FASTA1GB.fna"
## Source: https://www.ncbi.nlm.nih.gov/projects/genome/guide/human/index.shtml
if [ ${MODE} == 1 ]; then
	wget -nc "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_genomic.fna.gz" -O "${FASTA1GB}.gz"
else
	## Source: https://www.ebi.ac.uk/ena/browser/view/CEFR01
	wget -nc "ftp://ftp.ebi.ac.uk/pub/databases/ena/wgs/public/cef/CEFR01.fasta.gz" -O "${FASTA1GB}.gz"
fi
if test -f "${FASTA1GB}"; then
	echo "${FASTA1GB} already unzipped, skipping "
else
	echo "Unzipping ${FASTA1GB}"
	gzip -dkf "${FASTA1GB}.gz" > "${FASTA1GB}"
fi

# Download Fastq
FASTQ1GB="FASTQ1GB.fnq"
if [ ${MODE} == 1 ]; then
	## Source: https://www.ebi.ac.uk/ena/browser/view/SRX11174563?show=reads
	wget -nc "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR148/030/SRR14856430/SRR14856430_1.fastq.gz" -O "${FASTQ1GB}.gz"
else
	## Source: https://www.ebi.ac.uk/ena/browser/view/PRJNA602297?show=reads
	wget -nc "ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR109/049/SRR10916849/SRR10916849.fastq.gz" -O "${FASTQ1GB}.gz"
fi
if test -f "${FASTQ1GB}"; then
	echo "${FASTQ1GB} already unzipped, skipping "
else
	echo "Unzipping ${FASTQ1GB}"
	gzip -dkf "${FASTQ1GB}.gz" > "${FASTQ1GB}"
fi

cd ..
