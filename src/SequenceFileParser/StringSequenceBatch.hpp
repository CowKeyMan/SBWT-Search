#ifndef STRING_SEQUENCE_BATCHER_HPP
#define STRING_SEQUENCE_BATCHER_HPP

#include <string>
#include <vector>

using std::string;
using std::vector;

#include "Utils/TypeDefinitions.h"

namespace sbwt_search {

class StringSequenceBatch {
  public:
    vector<string> buffer;
    vector<u64> string_indexes;
    vector<u64> character_indexes;
};

}

#endif
