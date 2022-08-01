#ifndef SEQUENCE_FILE_PARSER_H
#define SEQUENCE_FILE_PARSER_H

/**
 * @file SequenceFileParser.h
 * @brief Contains functions for reading from FASTA or FASTQ files
 * */

#include <memory>
#include <string>
#include <vector>

#include "Utils/IOUtils.hpp"
#include "Utils/TypeDefinitions.h"
#include "kseq++/kseq++.hpp"
#include "kseq++/seqio.hpp"

using std::make_unique;
using std::unique_ptr;
using std::string;
using std::vector;
using klibpp::SeqStreamIn;

namespace sbwt_search {

class SequenceFileParser {
  private:
    string filename;
    u64 kmer_size;
    void add_sequence(const string seq);
    SeqStreamIn stream;

  public:
    SequenceFileParser(const string &filename, const u64 kmer_size):
        kmer_size(kmer_size),
        stream(filename.c_str()) {
      ThrowingIfstream::check_file_exists(filename);
    }
    string get_next();
    vector<string> get_all();
};

}

#endif
