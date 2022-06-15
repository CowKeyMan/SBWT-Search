#include <string>
#include <vector>
#include <zlib.h>

#include <kseq++/seqio.hpp>

#include "QueryReader.h"
#include "kseq++/kseq++.hpp"

using klibpp::KSeq;
using klibpp::SeqStreamIn;
using std::string;
using std::vector;

namespace sbwt_search {

QueryReader QueryReader::read() {
  KSeq record;
  SeqStreamIn stream(filename.c_str());
  while (stream >> record) {
    string read = record.seq;
    reads.push_back(read);
    total_letters += read.length();
  }
  total_positions = total_letters - kmer_size * reads.size() + 1 * reads.size();
  return *this;
}

}
