Searching
=========

The following diagram shows the search components. The diamonds show the disk output, the rectangles with sharp edges are the class components and the rectangles with round corners are the data objects which are passed around between components.

.. _SearchComponentsPipeline:
.. mermaid:: graphs/SearchComponentsPipeline.mmd

Note that the memory given are in bits

Main Memory
+++++++++++

In terms of main memory, we will use:

.. math::
   max\_characters * 64 * 7 max\_characters * 2 + max\_characters * 8 * 2 = max\_characters * 460 bits\ per\ batch

We must have at least 2 batches in our program, meaning that, given **1GB** of memory (:math:`(8 \times 10^9) bits`), we can use:

.. math::
    8 \times 10^9 / 460 / 2 = 8695652 characters per batch

* At 3 batches, then we could have at most *5797101 characters/batch*
* At 4 batches, then we could have at most *4347826 characters*
* At 5 batches, then we could have at most *3478260 characters*

Note that we will need to round down the number of characters to a multiple of 1024 so that they can be easily used in the GPU

VRAM
++++

We must also be wary of GPU memory. In VRAM we will only have a single batch at a time. We will have

.. math::

  max_characters * 64 + max_characters * 2 = max_characters * 66 bits

This means that for every 1GB of VRAM, we can have *121212121 characters*

**Thus, our searching algorithm uses 7x more RAM than VRAM**. However, please note that from our total VRAM, we must first remove the ammount occupied by the SBWT index itself.

Example of memory usage
+++++++++++++++++++++++

As an example:

* Say we have 40GB of GPU memory (VRAM)
* Our SBWT index takes 25GB of this memory
* This means that we are left with 15GB of VRAM
* To make the most effective use of the remaining VRAM, we should have 15 * 7 = 115GB or main memory (RAM) to make the most effective use of the available VRAM
* We need at least 2 batches, meaning that we would then need 230GB of main memory

Slight note: We should also have a few more bytes of RAM available for any extra small variables and other runtime objects that the program uses
