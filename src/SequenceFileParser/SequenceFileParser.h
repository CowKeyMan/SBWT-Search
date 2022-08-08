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

using klibpp::SeqStreamIn;
using std::make_unique;
using std::string;
using std::unique_ptr;
using std::vector;

namespace sbwt_search {

class SequenceFileParser {
  private:
    string filename;
    void add_sequence(const string seq);
    SeqStreamIn stream;
    bool reached_end = false;

  public:
    SequenceFileParser(const string &filename): stream(filename.c_str()) {
      ThrowingIfstream::check_file_exists(filename);
    }
    bool operator>>(string &s);
    vector<string> get_all();
};

}

#endif
