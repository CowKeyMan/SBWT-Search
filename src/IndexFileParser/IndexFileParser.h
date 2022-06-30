#ifndef INDEX_FILE_PARSER_H
#define INDEX_FILE_PARSER_H

/**
 * @file IndexFileParser.h
 * @brief Contains functions for interacting and reading SBWT index files
 * */

#include <memory>
#include <string>
#include <vector>

#include <sdsl/bit_vectors.hpp>

#include "IOUtils.hpp"
#include "Parser.h"
#include "TypeDefinitionUtils.h"

using sdsl::bit_vector;
using std::string;
using std::unique_ptr;
using std::vector;

namespace sbwt_search {

class IndexFileParser: Parser {
private:
  string filename;
  unique_ptr<bit_vector> sdsl_vector;
  vector<u64> custom_bit_vector;
  u64 *bit_vector_pointer;
  size_t bit_vector_size;
  size_t bits_total;
public:
  IndexFileParser(string filename): filename(filename) {
    check_file_exists(filename.c_str());
  }
  void parse_sdsl();
  void parse_c_bit_vector();
  void parse_sbwt_bit_vector();
  auto get_bit_vector_pointer() { return bit_vector_pointer; }
  auto get_bit_vector_size() { return bit_vector_size; }
  auto get_bits_total() { return bits_total; }
};

}

#endif
