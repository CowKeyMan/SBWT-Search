#ifndef STRING_SEQUENCE_BATCH_PRODUCER_H
#define STRING_SEQUENCE_BATCH_PRODUCER_H

/**
 * @file StringSequenceBatchProducer.h
 * @brief takes care of building and sending the stringsequencebatch
 */

#include <memory>
#include <stddef.h>
#include <string>

#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {
class StringSequenceBatch;
class ContinuousSequenceFileParser;
}  // namespace sbwt_search

using design_utils::SharedBatchesProducer;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

class StringSequenceBatchProducer:
  public SharedBatchesProducer<StringSequenceBatch> {
  friend ContinuousSequenceFileParser;

public:
  StringSequenceBatchProducer(uint max_batches);

private:
  void set_string(const string &s);
  shared_ptr<StringSequenceBatch> get_default_value() override;
};

}  // namespace sbwt_search
#endif
