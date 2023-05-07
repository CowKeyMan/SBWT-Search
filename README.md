# SBWT Search

An application for k-mer searching within an SBWT structure and also pseudoaligning genomes with GPU acceleration.

## For Users

This respository takes as inputs a themisto `.tdbg` and `.tcolors` structures created by themisto v3.0, which can be found at <https://github.com/algbio/themisto>.

### Requirements
REQUIREMENTS: gcc>=11.0, cuda/nvcc>=11.0

If you wish to use a lower version of cuda/nvcc, you may remove or comment out this line `add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=all>")` from the file `cmake/SetHipTargetDevice.cmake`

### Installation

To build the program, simply run the following commands:

```bash
./scripts/build/release.sh <NVIDIA|AMD|CPU>
```

Choose the platform for which you have a GPU for, or the CPU version if you do not have an NVIDIA or AMD GPU. Note that you will need to have CUDA installed, and if this is not installed, please check the [https://cowkeyman.github.io/SBWT-Search/documentation/Documents/WorkflowDocumentation/Tools.html](Tools) section in the documentation website for developers.

NOTE: The CPU version uses [https://github.com/ROCm-Developer-Tools/HIP-CPU](HIP-CPU) which is currently still in the development phase and has some bugs. It produced very strange results when it was executed on this repository.

Afterwards, an executable will be generated at `build/bin/sbwt_search`.

### Environment Variables

If you wish to see logs, set `SPDLOG_LEVEL=TRACE`. The values available to `SPDLOG_LEVEL` are, in order of verbosity, most verbose first: `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL`, `OFF`.

To set the number of threads used by the program, set `OMP_NUM_THREADS=<thread number>`. To use as many threads as you have cores (or twice that in the case of hyperthreading CPUs) you can run `unset OMP_NUM_THREADS`.

### Index Searching

The index searching is the first step of the pseudoalignment pipeline. It will search for the indexes of kmers within the SBWT structure and store them to disk. It was decided not to combine both index searching and color searching within the same step because some SBWT structures can be large and they might not fit in the GPU at the same time.

```
Usage:
  index [OPTION...]

  -q, --query-file arg          The query in FASTA or FASTQ format,
                                possibly gzipped, and also possibly a
                                combination of both. Empty lines are also
                                supported. If the file extension is
                                '.list', this is interpreted as a list of
                                query files, one per line. In this case,
                                --output-prefix must also be list of output
                                files in the same manner, one line for each
                                input file.
  -i, --index-file arg          The themisto *.tdbg file or SBWT's *.sbwt
                                file. The program is compatible with both.
                                This contains the 4 bit vectors for acgt as
                                well as the k for the k-mers used.
  -o, --output-prefix arg       The output file prefix or the output file
                                list. If the file ends with the extension
                                '.list', then it will be interepreted as a
                                list of file output prefixes, separated by
                                a newline. The extension of these output
                                files will be determined by the choice of
                                output format (look at the print-mode
                                option for more information chosen
  -u, --unavailable-main-memory arg
                                The amount of main memory not to consume
                                from the operating system in bits. This
                                means that the program will hog as much
                                main memory it can, provided that the VRAM
                                (GPU memory) can also keep up with it,
                                except for the amount specified by this
                                value. By default it is set to 1GB. The
                                value can be in the following formats:
                                12345B (12345 bytes), 12345KB, 12345MB or
                                12345GB (default: 8589934592)
  -m, --max-main-memory arg     The maximum amount of main memory (RAM)
                                which may be used by the searching step, in
                                bits. The default is that the program will
                                occupy as much memory as it can, minus the
                                unavailable main-memory. This value may be
                                skipped by a few megabytes for its
                                operation. It is only recommended to change
                                this when you have a few small queries to
                                process, so that intial memory allocation
                                is faster. The format of this value is the
                                same as that for the
                                unavailable-main-memory option (default:
                                18446744073709551615)
  -c, --cpu-memory-percentage arg
                                After calculating the memory usage using
                                the formula: 'min(system_max_memory,
                                max-memory) - unavailable-max-memory', we
                                multiply that value by memory-percentage,
                                which is this parameter. This parameter is
                                useful in case the program is unexpectedly
                                taking too much memory. By default it is
                                0.8, which indicates that 80% of available
                                memory will be used. Note, the total memory
                                used is not forced, and this is more of a
                                soft maximum. The actual memory used will
                                be slightly more for small variables and
                                other registers used throughout the
                                program. (default: 0.8)
  -r, --base-pairs-per-seq arg  The approximate number of base pairs in
                                every seq. This is necessary because we
                                need to keep track of the breaks where each
                                seq starts and ends in our list of base
                                pairs. As such we must allocate memory for
                                it. By defalt, this value is 100, meaning
                                that we would then allocate enough memory
                                for 1 break per 100 base pairs. This option
                                is available in case your seqs vary a lot
                                more than that and you wish to optimise for
                                space. (default: 100)
  -g, --gpu-memory-percentage arg
                                The percentage of gpu memory to use from
                                the remaining free memory after the index
                                has been loaded. This means that if we have
                                40GB of memory, and the index is 30GB, then
                                we have 10GB left. If this value is set to
                                0.9, then 9GB will be used and the last 1GB
                                of memory on the GPU will be left unused.
                                The default value is 0.95, and unless you
                                are running anything else on the machine
                                which is also GPU heavy, it is recommended
                                to leave it at this value. (default: 0.95)
  -s, --streams arg             The number of files to read and write in
                                parallel. This implies dividing the
                                available memory into <memory>/<streams>
                                pieces, so each batch will process less
                                items at a time, but there is instead more
                                parallelism. This should not be too high
                                nor too large, as the number of threads
                                spawned per file is already large, and it
                                also depends on your disk drive. The
                                default is 4. This means that 4 files will
                                be processed at a time. If are processing
                                less files than this, then the program will
                                automatically default to using as many
                                streams as you have files. (default: 4)
  -p, --print-mode arg          The mode used when printing the result to
                                the output file. Options are 'ascii'
                                (default), 'binary' or 'bool'. In ascii
                                mode the results will be printed in ASCII
                                format so that the number viewed represents
                                the position in the SBWT index. The indexes
                                within a seq are separated by spaces and
                                each seq is separated by a newline. Strings
                                which are not found are represented by -1
                                and strings which are invalid (they contain
                                characters other than ACGT) are represented
                                by a -2. For binary format, the output is
                                in binary, that is, each index takes 8
                                bits. The numbers are placed in a single
                                binary string where every 8 bytes
                                represents an unsigned 64-bit number.
                                Similarly to ASCII, strings which are not
                                found are represented by a -1 (which loops
                                around to become the maximum 64-bit integer
                                (ULLONG_MAX=18446744073709551615)), strings
                                which are invalid are represented by -2
                                (ULLONG_MAX-1) and seqs are separeted by a
                                -3 (ULLONG_MAX-2). This version turns out
                                to be slower and uses more space, it is
                                only recommended if your indexes are huge
                                (mostly larger than 8 bits). 'bool' is the
                                fastest mode however it is also the least
                                desriptive. In this mode, each index
                                results in a single ASCII byte, which
                                contains the value 0 if found, 1 if not
                                found and 2 if the value is invalid.
                                Similarly to the ascii format, each seq is
                                separated by a newline. This is the fastest
                                and most condensed way of printing the
                                results, but we lose the position in the
                                index, and therefore we cannot use this
                                format for pseudoalignment. In terms of
                                file extensions, ASCII format will add
                                '.txt', boolean format will add '.bool' and
                                binary format will add '.bin'. (default:
                                ascii)
  -k, --colors-file arg         The *.tcolors file produced by themisto
                                v3.0, which contains the key_kmer_marks as
                                one of its components within, used in this
                                program. When this option is given, then
                                the index search will move to the next key
                                kmer. If not given, then the program will
                                simply get the index of the node at which
                                the given k-mer lands on. (default: "")
      --no-headers              Do not write the headers to the outut
                                files. The headers are the format name and
                                version number written at the start of the
                                file. The format of these 2 strings are
                                that first we have a binary unsigned
                                integer indicating the size of the string
                                to come, followed by the string itself in
                                ascii format, which contains as many bytes
                                as indicated by the previous value. Then
                                another unsigned integer indicating the
                                size of the string representing the version
                                number, and then the string in ascii bytes
                                of the version number. By default this
                                option is false (meaning that the headers
                                WILL be printed by default). Please note
                                that if you wish to use the ascii or binary
                                format for pseudoalignment later, this
                                header is mandatory.
  -h, --help                    Print usage (you are here)
```

Within this repository, I included some nice test objects. You can run the following command and adjust it as you see fit to get yourself started:

```bash
./build/bin/sbwt_search index -q test_objects/search_test_indexed.fna -i test_objects/search_test_index.sbwt -o out -p ascii
```

You will then be able to see the output in out.txt, since our print-mode was ascii. Note: `search_test_index.sbwt` is the SBWT index generated for the file `search_test_indexes.fna`, and the numbers you see in out.txt will be the position of the kmer in the SBWT index. Note that while this file does not include any, we support files with a mixture of fasta and fastq files, as well as files with empty reads.

### Color Searching

After part 1 (index searching), we can run the color search on the results of this. The following are the command line parameters.

```
Usage:
  colors [OPTION...]

  -q, --query-file arg          Input file which is the output of the
                                previous step (index search). Note that we
                                cannot directly use the output from the
                                '--output-prefix' from the previous step
                                since we must also have the file extension
                                in this step. If the file extension is
                                '.list', this is interpreted as a list of
                                query files, one per line. In this case,
                                --output-prefix must also be list of output
                                files in the same manner, one line for each
                                input file.
  -o, --output-prefix arg       The output file prefix or the output file
                                list. If the file ends with the extension
                                '.list', then it will be interepreted as a
                                list of file output prefixes, separated by
                                a newline. The extension of these output
                                files will be determined by the choice of
                                output format (look at the print-mode
                                option for more information chosen
  -k, --colors-file arg         The *.tcolors file produced by themisto
                                v3.0, which contains the colors data used
                                in this program.
  -u, --unavailable-main-memory arg
                                The amount of main memory not to consume
                                from the operating system in bits. This
                                means that the program will hog as much
                                main memory it can, provided that the VRAM
                                (GPU memory) can also keep up with it,
                                except for the amount specified by this
                                value. By default it is set to 1GB. The
                                value can be in the following formats:
                                12345B (12345 bytes), 12345KB, 12345MB or
                                12345GB (default: 8589934592)
  -m, --max-main-memory arg     The maximum amount of main memory (RAM)
                                which may be used by the searching step, in
                                bits. The default is that the program will
                                occupy as much memory as it can, minus the
                                unavailable main-memory. This value may be
                                skipped by a few megabytes for its
                                operation. It is only recommended to change
                                this when you have a few small queries to
                                process, so that intial memory allocation
                                is faster. The format of this value is the
                                same as that for the
                                unavailable-main-memory option (default:
                                18446744073709551615)
  -c, --cpu-memory-percentage arg
                                After calculating the memory usage using
                                the formula: 'min(system_max_memory,
                                max-memory) - unavailable-max-memory', we
                                multiply that value by memory-percentage,
                                which is this parameter. This parameter is
                                useful in case the program is unexpectedly
                                taking too much memory. By default it is
                                0.8, which indicates that 80% of available
                                memory will be used. Note, the total memory
                                used is not forced, and this is more of a
                                soft maximum. The actual memory used will
                                be slightly more for small variables and
                                other registers used throughout the
                                program. (default: 0.8)
  -g, --gpu-memory-percentage arg
                                The percentage of gpu memory to use from
                                the remaining free memory after the index
                                has been loaded. This means that if we have
                                40GB of memory, and the index is 30GB, then
                                we have 10GB left. If this value is set to
                                0.9, then 9GB will be used and the last 1GB
                                of memory on the GPU will be left unused.
                                The default value is 0.95, and unless you
                                are running anything else on the machine
                                which is also GPU heavy, it is recommended
                                to leave it at this value. (default: 0.95)
  -p, --print-mode arg          The mode used when printing the result to
                                the output file. Options are 'ascii'
                                (default), 'binary' or 'csv'. In ascii
                                omde, the results are printed in ASCII
                                format so that the numbers viewed in each
                                line represent the colors found. The seqs
                                are separated by newline characters. The
                                binary format   will be similar to the
                                index search in that each color found will
                                be represented instead by an 8 byte
                                (64-bit) integer, and the start of a new
                                seq is indicated by a -1 (ULLONG_MAX). This
                                can result in larger files due to the
                                characters taken by the colors usually
                                being quite small, so the ascii format does
                                not take as many characters. The csv format
                                is the densest format and results in VERY
                                huge files. As such it is only recommended
                                to use it for smaller files. The format
                                consists of comma separated 0s and 1s,
                                where a 0 indicates that the color at that
                                index has not been found, while a 1
                                represents the opposite. (default: ascii)
  -s, --streams arg             The number of files to read and write in
                                parallel. This implies dividing the
                                available memory into <memory>/<streams>
                                pieces, so each batch will process less
                                items at a time, but there is instead more
                                parallelism. This should not be too high
                                nor too large, as the number of threads
                                spawned per file is already large, and it
                                also depends on your disk drive. The
                                default is 4. This means that 4 files will
                                be processed at a time. If are processing
                                less files than this, then the program will
                                automatically default to using as many
                                streams as you have files. (default: 4)
  -t, --threshold arg           The percentage of kmers within a seq which
                                need to be attributed to a color in order
                                for us to accept that color as being part
                                of our output. Must be a value between 1
                                and 0 (both included) (default: 1)
      --include-not-found       By default, indexes which have not been
                                found in the index search (represented by
                                -1s) are not considered by the algorithm,
                                and they are simply skipped over and
                                considered to not be part of the seq. If
                                this option is set, then they will be
                                considered as indexes which have had no
                                colors found.
      --include-invalid         By default, indexes which are invalid, that
                                is, the kmers to which they correspond to
                                contain characters other than acgt/ACGT
                                (represented by -2s) are not considered by
                                the algorithm, and they are simply skipped
                                over and considered to not be part of the
                                seq. If this option is set, then they will
                                be considered as indexes which have had no
                                colors found.
      --no-headers              Do not write the headers to the outut
                                files. The headers are the format name and
                                version number written at the start of the
                                file. The format of these 2 strings are
                                that first we have a binary unsigned
                                integer indicating the size of the string
                                to come, followed by the string itself in
                                ascii format, which contains as many bytes
                                as indicated by the previous value. Then
                                another unsigned integer indicating the
                                size of the string representing the version
                                number, and then the string in ascii bytes
                                of the version number. For the csv version,
                                this header is a comma separated list of
                                color ids at the first line of the file. By
                                default this option is false (meaning that
                                the headers WILL be printed by default).
                                Please note that if you wish to use the
                                ascii or binary format for pseudoalignment
                                later, this header is mandatory.
  -r, --indexes-per-seq arg     The approximate number of indexes in every
                                seq. This is necessary because we need to
                                keep track of the breaks where each seq
                                starts and ends in our list of base pairs.
                                As such we must allocate memory for it. By
                                defalt, this value is 70, meaning that we
                                would then allocate enough memory for 1
                                break per 70 indexes. This option is
                                available in case your seqs vary a lot more
                                than that and you wish to optimise for
                                space. (default: 70)
  -h, --help                    Print usage (you are here)
```

As a sample for this, you may run the following:

```bash
# download the larger test objects from dropbox, because they are too big to put in the git repository
./scripts/configure/download_large_test_objects.sh
./build/bin/sbwt_search colors -q test_objects/themisto_example/GCA_queries.txt -k test_objects/themisto_example/GCA_combined_d1.tcolors -o out -p ascii -t 0.7
```

You will then see the colors printed in out.txt, since our print-mode was ascii. Note that this part also supports empty lines.

## For Developers

The documentation for developing this code base lies in following website: <https://cowkeyman.github.io/SBWT-Search>. The pages are built using the documentation of the repository itself using github actions.

## Credits and Licenses

This repository makes use of the following papers:

* Zhou, D., Andersen, D. G., & Kaminsky, M. (2013). Space-Efficient, High-Performance Rank and Select Structures on Uncompressed Bit Sequences. In V. Bonifaci, C. Demetrescu, & A. Marchetti-Spaccamela (Eds.), Experimental Algorithms (pp. 151–163). Springer Berlin Heidelberg.
* Alanko, J. N., Puglisi, S. J., & Vuohtoniemi, J. (2022). Succinct k-mer Set Representations Using Subset Rank Queries on the Spectral Burrows-Wheeler Transform (SBWT). BioRxiv. https://doi.org/10.1101/2022.05.19.492613
* GPU Index Searching function by Harri Kähkönen, University of Helsinki (Master's thesis in progress)

Furthermore, we make use of code and ideas from other code repositories:

* The CUDA search function is based on this repository: <https://version.helsinki.fi/harrikah/gpu-computing>
* The API is based on this repository: <https://github.com/algbio/SBWT>

The repository also makes use of a number of other tools/code bases, however we do not distribute these as part of our code base. Instead they are downloaded automatically using CMake, or are shown in the tools section of the github pages documentation.

## TODO

* Currently I am not making the best use of RAM when reserving memory, and I can have batch sizes be larger
* Support for writing zipped indexes and colors
* Support for reading zipped indexes in the colors phase
* Static binaries
* Project documentation in the developer documentation
