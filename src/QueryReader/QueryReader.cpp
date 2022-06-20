#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

#include <kseq++/seqio.hpp>

#include "QueryReader.h"
#include "kseq++/kseq++.hpp"

using klibpp::KSeq;
using klibpp::make_kstream;
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
    add_sequence(seq);
  }
}

auto QueryReader::add_sequence(const string &seq) -> void {
  seqs.push_back(seq);
  total_letters += seq.length();
  if (seq.length() >= kmer_size) {
    total_positions += seq.length() - kmer_size + 1;
  }
}

auto QueryReader::parse_kseqpp_read() -> void {
  check_if_has_parsed();
  auto iss = SeqStreamIn(filename.c_str());
  auto records = iss.read();
  for (auto &record: records) {
    string seq = record.seq;
    add_sequence(seq);
  }
}

auto QueryReader::parse_kseqpp_gz_stream() -> void {
  check_if_has_parsed();
  KSeq record;
  gzFile fp = gzopen(filename.c_str(), "r");
  auto stream = make_kstream(fp, gzread, klibpp::mode::in);
  while (stream >> record) { add_sequence(record.seq); }
  gzclose(fp);
}

}
