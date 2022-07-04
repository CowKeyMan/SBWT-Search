Components
==========

This section will describe the components which make up our code base such that it is easier to navigate it.


The project code is located int

* **src**: The folder within which the source code is located

  * `main.cpp`: Main executable from which everything else is used
  * `benchmark_main.cpp`: Benchmarks different implementations of our components
  * `test_main.cpp`: Main executable for tests. Searches for all files ending in *_test.cpp* and adds them to its tests

  * **ArgumentParser**: Contains tools to parse command line arguments
  * **IndexFileParser**: Parses the SBWT index files
  * **Parser**: Super class for parsers
  * **QueryFileParser**: Parses the query files in FASTA or FASTQ format
  * **RankIndexBuilder**: Builds the rank index of the SBWT index files
  * **RawSequencesParser**: Transoforms the raw FASTA/FAST sequences into their binary counterpart
  * **Utils**: Contains a collection of helpful utilities

Hierarchy
+++++++++

.. _MainPipeline:
.. mermaid:: graphs/Classhierarchy.mmd
