#ifndef INDEX_FILE_PARSER_TEST_UTILS_H
#define INDEX_FILE_PARSER_TEST_UTILS_H

/**
 * @file IndexFileParserTestUtils.h
 * @brief Methods used by the testing modules of the IndexFileParsers
 */

#include <string>
#include <vector>

namespace sbwt_search {

using std::string;
using std::vector;

[[nodiscard]] auto get_binary_index_output_filename() -> string;

auto write_fake_binary_results_to_file() -> void;

}  // namespace sbwt_search

#endif
