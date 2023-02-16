#ifndef STRING_SEQUENCE_BATCH_PRODUCER_H
#define STRING_SEQUENCE_BATCH_PRODUCER_H

/**
 * @file StringSequenceBatchProducer.h
 * @brief takes care of building and sending the stringsequencebatch
 */

#include <memory>
#include <string>

#include "BatchObjects/StringSequenceBatch.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

class ContinuousSequenceFileParser;

using design_utils::SharedBatchesProducer;
using std::shared_ptr;
using std::string;

class StringSequenceBatchProducer:
    public SharedBatchesProducer<StringSequenceBatch> {
  using Base = SharedBatchesProducer<StringSequenceBatch>;
  friend ContinuousSequenceFileParser;

public:
  explicit StringSequenceBatchProducer(u64 max_batches);

private:
  void set_string(const string &s);
  auto get_default_value() -> shared_ptr<StringSequenceBatch> override;
};

}  // namespace sbwt_search
#endif
