# SBWT Search

An application to search for k-mers in a genome given an SBWT index (which indexes k-mers of genomes)

## For Users

This respository takes 2 inputs: An SBWT index which can be produced by using this repository: <https://github.com/algbio/SBWT>, and a FASTA/FASTQ file to search within the SBWT index.

To build the program, simply run the following commands:

```bash
./scripts/build/release.sh a
```

Note, you may be missing some depenencies, in which case follow the instructions or errors given by CMake. A common dependency is libgz, which can be downloaded using the following command: `sudo apt install zlib1g-dev`

Afterwards, an executable will be generated at `build/bin/main`. You can run this and a help text will appear. Here is the API of this binary:

```
Usage:
  SBWT_Search [OPTION...]

  -o, --output-file arg         Output filename
  -i, --index-file arg          Index input file
  -q, --query-file arg          The query in FASTA or FASTQ format,
                                possibly gzipped.Multi-line FASTQ is not
                                supported. If the file extension is .txt,
                                this is interpreted as a list of query
                                files, one per line. In this case,
                                --out-file is also interpreted as a list of
                                output files in the same manner, one line
                                for each input file.
  -u, --unavailable-main-memory arg
                                The amount of main memory not to consume
                                from the operating system in bits. This
                                means that the program will hog as much
                                main memory it can, provided that the VRAM
                                can also keep up with it, except for the
                                amount specified by this value. By default
                                it is set to 4GB. The value can be in the
                                following formats: 12345 (12345 bits),
                                12345B (12345 bytes), 12345KB, 12345MB or
                                12345GB (default: 34359738368)
  -m, --max-main-memory arg     The maximum amount of main memory (RAM)
                                which may be used by the searching step, in
                                bits. The default is that the program will
                                occupy as much memory as it can, minus the
                                unavailable main-memory. This value may be
                                skipped by a few megabytes for its
                                operation. It is only recommended to change
                                this when you have a few small queries to
                                process. The format of this value is the
                                same as that for the
                                unavailable-main-memory option (default:
                                18446744073709551615)
  -b, --batches arg             The number of batches to use. The default
                                is 5. 1 is the minimum, and is equivalent
                                to serial processing in terms of speed.
                                This will split the main memory between the
                                components. The more batches, the lower
                                that a single batch's size. 5 is the
                                recommended because there are 5 components
                                so they can all keep processing without
                                interruption from the start (this is
                                assuming you have 5 threads running). If
                                you have less threads, maybe set to to the
                                number of available threads instead
                                (default: 5)
  -c, --print-mode arg          The mode used when printing the result to
                                the output file. Options are 'ascii'
                                (default), 'binary' or 'boolean'. In ascii
                                mode the results will be printed in ASCII
                                format so that the number viewed output
                                represents the position in the SBWT index.
                                The outputs are separated by spaces and
                                each word is separated by a newline.
                                Strings which are not found are represented
                                by -1 and strings which are invalid are
                                represented by a -2. For binary format, the
                                output is in binary. The numbers are placed
                                in a single binary string where every 8
                                bytes represents an unsigned 64-bit number.
                                Similarly to ASCII, strings which are not
                                found are represented by a -1 (which loops
                                around to become the maximum 64-bit integer
                                (ULLONG_MAX=18446744073709551615)), strings
                                which are invalid are represented by -2
                                (ULLONG_MAX-1) and strings are separeted by
                                a -3 (ULLONG_MAX-2). The binary version is
                                much faster but requires decoding the file
                                later when it needs to be viewed. 'boolean'
                                is the fastest mode however it is also the
                                least desriptive. In this mode, 2 files are
                                output. The first file is named by the
                                given output file name, and contains 1 bit
                                for each result. The string sizes are given
                                in another file, where every 64 bit integer
                                here is a string size. This is the fastest
                                and most condensed way of printing the
                                results, but we lose some information
                                because we cannot say wether the result is
                                invalid or just not found. At the end of
                                this data file, the final number is padded
                                by 0s to the next 64-bit integer. The
                                second file is called
                                <output_filename>_seq_sizes. Here, every 64
                                bit binary integer represents the amount of
                                results for each string in the original
                                input file. (default: ascii)
  -h, --help                    Print usage
```

Within this repository, I included some nice test objects. You can run the following command and adjust it as you see fit to get yourself started:

```sh
./build/bin/main_cuda -q test_objects/search_test_indexed.fna -i test_objects/search_test_index.sbwt -o out.txt -b 5 -c ascii
```

You will then be able to see the output in out.txt. Note: `search_test_index.sbwt` is the SBWT index generated for the file `search_test_indexes.fna`, and the numbers you see in out.txt will be the position of the kmer in the SBWT index.

If you wish to see the logs, you can run `export SPDLOG_LEVEL=TRACE`.

## For Developers

The documentation for developing this code base lies in the github pages: <https://cowkeyman.github.io/CPP_github_workflow/>. The pages are built using the documentation of the repository itself using gitgub actions.

## Credits and Licenses

This repository makes use of the following papers:

* Zhou, D., Andersen, D. G., & Kaminsky, M. (2013). Space-Efficient, High-Performance Rank and Select Structures on Uncompressed Bit Sequences. In V. Bonifaci, C. Demetrescu, & A. Marchetti-Spaccamela (Eds.), Experimental Algorithms (pp. 151–163). Springer Berlin Heidelberg.
* Alanko, J. N., Puglisi, S. J., & Vuohtoniemi, J. (2022). Succinct k-mer Set Representations Using Subset Rank Queries on the Spectral Burrows-Wheeler Transform (SBWT). BioRxiv. https://doi.org/10.1101/2022.05.19.492613
* GPU Searching function by Harri Kähkönen, University of Helsinki (Master's thesis in progress)

Furthermore, we make use of code and ideas from other code repositories:

* The CUDA search function is based on this repository: <https://version.helsinki.fi/harrikah/gpu-computing>
* The API is based on this repository: <https://github.com/algbio/SBWT>

The repository also makes use of a number of other tools/code bases, however we do not distribute these as part of our code base. Instead they are downloaded automatically using CMake, or are shown in the tools section of the github pages documentation.
