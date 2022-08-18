#ifndef SEQUENCE_FILE_PARSER_H
#define SEQUENCE_FILE_PARSER_H

/**
 * @file SequenceFileParser.h
 * @brief Contains functions for reading from FASTA or FASTQ files
 * */

#include <memory>
#include <string>
#include <vector>

#include <kseq++/seqio.hpp>

#include "Utils/IOUtils.hpp"
#include "Utils/TypeDefinitions.h"

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
    SequenceFileParser(const string &filename);
    bool operator>>(string &s);
    vector<string> get_all();
};

}

#endif
