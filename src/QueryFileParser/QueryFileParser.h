#ifndef QUERY_FILE_PARSER_H
#define QUERY_FILE_PARSER_H

/**
 * @file QueryFileParser.h
 * @brief Contains functions for reading from FASTA or FASTQ files
 * */

#include <string>
#include <vector>

#include "Builder/Builder.h"
#include "Utils/IOUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::string;
using std::vector;

namespace sbwt_search {

class QueryFileParser: Builder {
  private:
    string filename;
    u64 kmer_size;
    void add_sequence(const string &seq);
    vector<string> seqs;
    u64 total_letters = 0;
    u64 total_positions = 0;

  public:
    QueryFileParser(const string &filename, const u64 kmer_size):
        filename(filename), kmer_size(kmer_size) {
      ThrowingIfstream::check_file_exists(filename.c_str());
    }
    auto &get_seqs() { return seqs; };
    auto get_total_letters() { return total_letters; };
    auto get_total_positions() { return total_positions; };
    void parse_kseqpp_streams();
    void parse_kseqpp_read();
    void parse_kseqpp_gz_stream();
    void parse_kseqpp_gz_read();
};

}

#endif
