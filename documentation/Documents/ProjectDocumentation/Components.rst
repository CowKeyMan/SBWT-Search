Components
==========

This section will describe the components which make up our code base such that it is easier to navigate it. The header files contain a docstring on top which briefly tells what each module is responsible of. I recommend looking at this when you need even more details

Hierarchy
+++++++++

We will first show the class Hierarchy and then proceed to tell how each component works

.. _MainPipeline:
.. mermaid:: graphs/Classhierarchy.mmd

The project code is located in the **src** folder.

* **src**: The folder within which the source code is located

  * `test_main.cpp`: Main executable for tests. Searches for all files ending in *_test.cpp* and adds them to its tests
  * `verify.cpp`: Verify that outputs of the main program contain the same contents. Check its help section for usage
  * `main.cpp`: Main executable from which everything else is used

  * **ArgumentParser**: Contains tools to parse command line arguments
  * **SbwtBuilder**: Parses the SBWT index files and builds a SbwtContainer from it
  * **SbwtContainer**: Contains the entire information about our index. This includes the plain-matrix vectors, the kmer size, the C vector/map and the rank index / `Poppy Data Structure <https://www.cs.cmu.edu/~dga/papers/zhou-sea2013.pdf>`_
  * **Presearcher**: Does searching for the Poppy data structure on the GPU and stores it in the SbwtContainer
  * **FilenamesParser**: Gets filenenames and checks if they are a single file or if they contain multiple files within them. If the ending of the file is '.txt', when it will expect that the given files are a list of other files.
  * **SequenceFileParser**: Reads the input files and extracts necessary data
  * **SeqToBitsConverter**: Converts sequences of characters to a bit vector (where each character takes 2 bytes)
  * **PositionsBuilder**: Builds the positions vector to be used for searching. Each kmer of size k gerts a position
  * **Searcher**: Searches for the kmers indicated by the positions and bit vectors in the Sbwt Index
  * **ResultsPrinter**: Outputs the results to disk. This class has 3 different variants/subclasses that the user can choose from. Check the help menu of the main executable for more information on these.

There are a couple more folders and components in the project, however the above should give a very good introduction to any programmer who whishes to know how to get started with the project.
