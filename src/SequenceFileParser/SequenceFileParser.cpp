#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <zlib.h>

#include <kseq++/seqio.hpp>

#include "SequenceFileParser/SequenceFileParser.h"
#include "kseq++/kseq++.hpp"

using klibpp::KSeq;
using klibpp::make_ikstream;
using klibpp::make_kstream;
using klibpp::SeqStreamIn;
using std::string;
using std::vector;

namespace sbwt_search {

auto SequenceFileParser::add_sequence(const string &seq) -> void {
  total_letters += seq.length();
  if (seq.length() >= kmer_size) {
    total_positions += seq.length() - kmer_size + 1;
  }
  seqs->push_back(std::move(seq));
}

auto SequenceFileParser::parse_kseqpp_streams() -> void {
  check_if_has_built();
  KSeq record;
  SeqStreamIn stream(filename.c_str());
  while (stream >> record) { add_sequence(record.seq); }
}

auto SequenceFileParser::parse_kseqpp_read() -> void {
  check_if_has_built();
  auto iss = SeqStreamIn(filename.c_str());
  auto records = iss.read();
  for (auto &record: records) { add_sequence(record.seq); }
}

auto SequenceFileParser::parse_kseqpp_gz_stream() -> void {
  check_if_has_built();
  KSeq record;
  gzFile fp = gzopen(filename.c_str(), "r");
  auto stream = make_kstream(fp, gzread, klibpp::mode::in);
  while (stream >> record) { add_sequence(record.seq); }
  gzclose(fp);
}

auto SequenceFileParser::parse_kseqpp_gz_read() -> void {
  check_if_has_built();
  gzFile fp = gzopen(filename.c_str(), "r");
  auto records = make_ikstream(fp, gzread).read();
  gzclose(fp);
  for (auto &record: records) { add_sequence(record.seq); }
}

}
