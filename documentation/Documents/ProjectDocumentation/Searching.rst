Searching
=========

The following diagram shows the search components. The diamonds show the items stored on disk, the rectangles with sharp edges are the class components and the rectangles with round corners are the data objects which are passed around between components.

.. _SearchComponentsPipeline:
.. mermaid:: graphs/SearchComponentsPipeline.mmd

Main Memory
+++++++++++

In terms of main memory, we will use:

.. math::
   max\_characters * (64 * 2 + 8 * 2 + 2) = max\_characters * 146) bits\ per\ batch

We also have chars_before_newline and newlines_before_newfile to store, however these are usually very small vectors if the fasta and fastq files are well formatted. To be sure we do not run into any problems, we should probably reserve 5 percent of available memory for them.

This means that given **1GB** of memory (:math:`(8 \times 1024^3) bits`), we can use:

.. math::
    8 \times 1024^3 / 146 = 58,835,168 characters per batch

* At 2 simultaneous batches, we could have at most *29,417,584 characters/batch*
* At 5 batches, then we could have at most *11,767,033 characters*

We should aim for 5 parallel batches, since we have 5 components which can run in parallel in our pipeline

Note that we will need to round down the number of characters to a multiple of 1024 so that they can be easily used in the GPU

VRAM
++++

We must also be wary of GPU memory. In VRAM we will only have a single batch at a time. We will have

.. math::

  max_characters * 64 + max_characters * 2 = max_characters * 66 bits

This means that for every 1GB of VRAM, we can have *121,212,121 characters*

**Thus, our searching algorithm uses 2.21x more RAM than VRAM** for every batch. Ideally, we have 11.5x more free RAM than VRAM.

However, please note that from in VRAM, we must also store the SBWT index, so we will have much less of this available usually.

Example of memory usage
+++++++++++++++++++++++

As an example:

* Say we have 40GB of GPU memory (VRAM)
* Our SBWT index takes 25GB of this memory
* This means that we are left with 15GB of VRAM
* To make the most effective use of the remaining VRAM, we should have 15 * 11.5 = 172GB of main memory (RAM) to make the most effective use of the available VRAM

NOTE: We should also have a few more bytes of RAM available (about 5% more should be more than enough for well formed FASTA/FASTQ files) for any extra small variables and other runtime objects that the program uses

* By 'well formed FASTA/FASTQ files' we mean that each sequence is not very short (more than 20 characters long) and there are not many empty lines.

* The 5% (actually set to 6% for extra peace of mind) is taken care of within the codebase, no need for the user to take care of anything
