#include <ctime>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>
#include <zlib.h>

#include <kseq++/seqio.hpp>

#include "SequenceFileParser/SequenceFileParser.h"
#include "kseq++/kseq++.hpp"

using klibpp::KSeq;
using std::move;
using std::out_of_range;
using std::string;
using std::vector;

namespace sbwt_search {

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

}
