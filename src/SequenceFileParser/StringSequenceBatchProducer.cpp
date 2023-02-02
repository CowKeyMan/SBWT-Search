#include <memory>
#include <string>

#include "BatchObjects/StringSequenceBatch.h"
#include "SequenceFileParser/StringSequenceBatchProducer.h"
#include "Tools/TypeDefinitions.h"

using std::make_shared;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

StringSequenceBatchProducer::StringSequenceBatchProducer(uint max_batches):
  Base(max_batches) {
  initialise_batches();
}

auto StringSequenceBatchProducer::get_default_value()
  -> shared_ptr<StringSequenceBatch> {
  return make_shared<StringSequenceBatch>();
}

auto StringSequenceBatchProducer::set_string(const string &s) -> void {
  get_batches().current_write()->seq = &s;
}

}  // namespace sbwt_search
