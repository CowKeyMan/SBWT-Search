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
  vector<string> seqs;
  u64 total_letters = 0;
  u64 total_positions = 0;
  uint kmer_size;
  bool has_parsed = false;
  void check_if_has_parsed();
  void add_sequence(const string &seq);

public:
  QueryReader(const string &filename, const uint kmer_size):
    filename(filename), kmer_size(kmer_size){};
  const vector<string> &get_seqs() { return seqs; };
  const size_t &get_total_letters() { return total_letters; };
  const size_t &get_total_positions() { return total_positions; };
  // Different types of seqs for benchmarking purposes
  // The clean way to do this would be to use virtual functions
  // But we want to avoid virtual function calls
  void parse_kseqpp_streams();
  void parse_kseqpp_read();
};

}

#endif
