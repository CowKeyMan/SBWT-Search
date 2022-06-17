#include <stdexcept>
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

auto QueryReader::check_if_has_parsed() -> void {
  if (has_parsed) {
    throw std::logic_error("QueryReader has already parsed a file");
  }
  has_parsed = true;
}

auto QueryReader::parse_kseqpp_streams() -> void {
  check_if_has_parsed();
  KSeq record;
  SeqStreamIn stream(filename.c_str());
  while (stream >> record) {
    string seq = record.seq;
    seqs.push_back(seq);
    total_letters += seq.length();
  }
  total_positions = total_letters - kmer_size * seqs.size() + 1 * seqs.size();
}

auto QueryReader::parse_kseqpp_read() -> void {
  check_if_has_parsed();
  auto iss = SeqStreamIn(filename.c_str());
  auto records = iss.read();
  for (auto &record: records) {
    string seq = record.seq;
    seqs.push_back(seq);
    total_letters += seq.length();
  }
  total_positions = total_letters - kmer_size * seqs.size() + 1 * seqs.size();
}

}
