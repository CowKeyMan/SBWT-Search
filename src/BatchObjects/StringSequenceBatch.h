#ifndef STRING_SEQUENCE_BATCH_H
#define STRING_SEQUENCE_BATCH_H

/**
 * @file StringSequenceBatch.h
 * @brief Data class for the string sequences and associated data. The buffer
 * stores the strings themselves. the buffers are then split into some number of
 * readers. Thus, the rest are indexes. string_indexes is the index of the
 * string itself, char_indexes are the indexes of the character within the
 * string while cumulative_char_indexes are the global index of the character.
 * So if we had the strings: "ABCD", "EF" and "GHI" in the buffer, and we wanted
 * to divide this by 3 readers with 3 characters per reader, then we would have:
 * string_indexes = {0, 0, 2, 3}, char_indexes = {0, 3, 0, 0},
 * cumulative_char_indexes = {0, 3, 6, 9}. Notice how at the end of
 * string_indexes and cumulative_char_indexes we always put the maximum, while
 * we put a 0 at the end of char_indexes. If we had 4 readers with 3 characters
 * per reader, we would have: string_indexes = {0, 0, 2, 3, 3}, char_indexes =
 * {0, 3, 0, 0, 0}, cumulative_char_indexes = {0, 3, 6, 9, 9}. Notice how if we
 * have more readers than characters to give it, we simply duplicate the maximum
 * or 0s at the end.
 * */

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
    vector<u64> char_indexes;
    vector<u64> cumulative_char_indexes;
};

}  // namespace sbwt_search

#endif
