#include <tuple>
#include <memory>
#include <stdexcept>

#include "SequenceFileParser/ContinuousSequenceFileParser.h"
#include "Utils/TypeDefinitions.h"

using std::tuple;
using std::shared_ptr;

namespace sbwt_search {

auto ContinuousSequenceFileParser::read() -> void {
  reader.read();
  finished_reading = true;
  free_all_consumers();
}

auto ContinuousSequenceFileParser::free_all_consumers() -> void {
  for (int i = 0; i < readers_amount; ++i) { character_semaphore.release(); }
}

bool ContinuousSequenceFileParser::operator>>(
  tuple<shared_ptr<vector<string>> &, u64 &, u64 &> t
) {
  return writer >> t;
}

}
