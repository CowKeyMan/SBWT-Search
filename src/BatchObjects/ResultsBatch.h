#ifndef RESULTS_BATCH_H
#define RESULTS_BATCH_H

/**
 * @file ResultsBatch.h
 * @brief Contains the vector with the results obtained after searching for the
 * k-mer within the SBWT index
 */

#include <vector>

using std::vector;

class ResultsBatch {
  public:
    vector<size_t> results;
};

#endif
