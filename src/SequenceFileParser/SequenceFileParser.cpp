#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <kseq++/kseq++.hpp>
#include <kseq++/seqio.hpp>

#include "SequenceFileParser/SequenceFileParser.h"
#include "Utils/IOUtils.hpp"

using klibpp::KSeq;
using std::move;
using std::out_of_range;
using std::string;
using std::vector;

namespace sbwt_search {

SequenceFileParser::SequenceFileParser(const string &filename):
    stream(filename.c_str()) {
  ThrowingIfstream::check_file_exists(filename);
}

auto SequenceFileParser::get_all() -> vector<string> {
  auto seqs = vector<string>();
  KSeq record;
  while (stream >> record) { seqs.push_back(move(record.seq)); }
  return seqs;
}

auto SequenceFileParser::operator>>(string &s) -> bool {
  if (stream.eof()) { return false; }
  KSeq record;
  stream >> record;
  s = move(record.seq);
  return true;
}

}  // namespace sbwt_search
