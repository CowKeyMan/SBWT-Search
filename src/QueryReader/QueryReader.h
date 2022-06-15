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
  vector<string> reads;
  u64 total_letters = 0;
  u64 total_positions = 0;
  uint kmer_size;

public:
  QueryReader(const string filename, const uint kmer_size):
    filename(filename), kmer_size(kmer_size){};
  QueryReader read();
  const vector<string> get_reads() { return reads; };
  const size_t get_total_letters() { return total_letters; };
  const size_t get_total_positions() { return total_positions; };
};

}

#endif
