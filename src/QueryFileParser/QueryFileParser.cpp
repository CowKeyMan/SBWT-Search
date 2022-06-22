#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <zlib.h>

#include <kseq++/seqio.hpp>

#include "QueryFileParser.h"
#include "kseq++/kseq++.hpp"

using klibpp::KSeq;
using klibpp::make_ikstream;
using klibpp::make_kstream;
using klibpp::SeqStreamIn;
using std::string;
using std::vector;

namespace sbwt_search {

auto QueryFileParser::check_if_has_parsed() -> void {
  if (has_parsed) {
    throw std::logic_error("QueryFileParser has already parsed a file");
  }
  has_parsed = true;
}

auto QueryFileParser::add_sequence(const string &seq) -> void {
  seqs.push_back(std::move(seq));
  total_letters += seq.length();
  if (seq.length() >= kmer_size) {
    total_positions += seq.length() - kmer_size + 1;
  }
}

auto QueryFileParser::parse_kseqpp_streams() -> void {
  check_if_has_parsed();
  KSeq record;
  SeqStreamIn stream(filename.c_str());
  while (stream >> record) { add_sequence(record.seq); }
}

auto QueryFileParser::parse_kseqpp_read() -> void {
  check_if_has_parsed();
  auto iss = SeqStreamIn(filename.c_str());
  auto records = iss.read();
  for (auto &record: records) { add_sequence(record.seq); }
}

auto QueryFileParser::parse_kseqpp_gz_stream() -> void {
  check_if_has_parsed();
  KSeq record;
  gzFile fp = gzopen(filename.c_str(), "r");
  auto stream = make_kstream(fp, gzread, klibpp::mode::in);
  while (stream >> record) { add_sequence(record.seq); }
  gzclose(fp);
}

auto QueryFileParser::parse_kseqpp_gz_read() -> void {
  check_if_has_parsed();
  gzFile fp = gzopen(filename.c_str(), "r");
  auto records = make_ikstream(fp, gzread).read();
  gzclose(fp);
  for (auto &record: records) { add_sequence(record.seq); }
}

}
