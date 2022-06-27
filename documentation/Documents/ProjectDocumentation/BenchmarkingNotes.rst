Benchmarking Notes
==================

Kseqpp
++++++

* Using streams with `std::move` provided excellent results. `std::move` sped up the process by around 1/3. The gz version without `std::move` took around the same time for the FASTA benchmark, so my intuition is that this function does `std::move` implicitly.
*
