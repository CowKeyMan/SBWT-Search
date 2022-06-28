#ifndef INDEX_FILE_PARSER
#define INDEX_FILE_PARSER

/**
 * @file IndexFileParser.h
 * @brief Contains functions for interacting and reading SBWT index files
 * */

#include <string>
#include <vector>
#include <memory>
#include <sdsl/bit_vectors.hpp>
#include "TypeDefinitionUtils.h"
#include "Parser.h"

using std::string;
using std::vector;
using std::unique_ptr;
using sdsl::bit_vector;

namespace sbwt_search {

class IndexFileParser: Parser {
private:
  string filename;
  unique_ptr<bit_vector> sdsl_vector;
  vector<u64> custom_bit_vector;
  u64* bit_vector_pointer;
  size_t bit_vector_size;
public:
  IndexFileParser(string filename): filename(filename) {}
  void parse_sdsl();
  void parse_bit_vectors();
  auto get_bit_vector_pointer() { return bit_vector_pointer; }
  auto get_bit_vector_size() { return bit_vector_size; }
};

}

#endif
