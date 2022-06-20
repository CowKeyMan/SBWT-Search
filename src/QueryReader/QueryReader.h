#ifndef QUERY_READER_H
#define QUERY_READER_H

/**
 * @file QueryReader.h
 * @brief Contains functions for reading from FASTA or FASTQ files
 * */

#include <string>
#include <vector>

#include "GlobalDefinitions.h"

using std::string;
using std::vector;

namespace sbwt_search {

class QueryReader {
private:
  string filename;
  uint kmer_size;
  bool has_parsed = false;
  void check_if_has_parsed();
  void add_sequence(const string &seq);
  vector<string> seqs;
  u64 total_letters = 0;
  u64 total_positions = 0;

public:
  QueryReader(const string &filename, const uint kmer_size):
    filename(filename), kmer_size(kmer_size){};
  auto &get_seqs() { return seqs; };
  auto get_total_letters() { return total_letters; };
  auto get_total_positions() { return total_positions; };
  void parse_kseqpp_streams();
  void parse_kseqpp_read();
};

}

#endif
