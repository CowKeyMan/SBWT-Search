#ifndef SEQ_STATISTICS_BATCH_H
#define SEQ_STATISTICS_BATCH_H

/**
 * @file SeqStatisticsBatch.h
 * @brief Stores statistics about each sequence of indexes. These include how
 * many ids were actually found within this seq, how many were invalid and how
 * many were not found. Each of these are vectors, since a single batch can have
 * many seqs. A seq may be split between 2 batches, so we also also store a
 * boolean if the first seq of this batch should be joined with the last seq of
 * the previous batch. We also have a colored_seq_id, which is useful since we
 * do not store '0' colors for warps and seqs without colors, so we keep a
 * cumulative id here for the colors id, where the colors are stored in a
 * separate batch object. The seqs_before_new_file stores how many sequences we
 * need to process before we need to start considering the next sequences as
 * part of the next file.
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class SeqStatisticsBatch {
public:
  vector<u64> found_idxs;
  vector<u64> invalid_idxs;
  vector<u64> not_found_idxs;
  vector<u64> colored_seq_id;
  vector<u64> seqs_before_new_file;

  auto reset() -> void {
    found_idxs.resize(0);
    invalid_idxs.resize(0);
    not_found_idxs.resize(0);
    colored_seq_id.resize(0);
    seqs_before_new_file.resize(0);
  }
};

}  // namespace sbwt_search

#endif
