#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <zlib.h>

#include <kseq++/seqio.hpp>

#include "SequenceFileParser/SequenceFileParser.h"
#include "kseq++/kseq++.hpp"

using klibpp::KSeq;
using std::move;
using std::string;
using std::vector;
using std::out_of_range;

namespace sbwt_search {

auto SequenceFileParser::get_all() -> vector<string> {
  auto seqs = vector<string>();
  KSeq record;
  while (stream >> record) { seqs.push_back(move(record.seq)); }
  return seqs;
}

auto SequenceFileParser::get_next() -> string {
  KSeq record;
  if (stream >> record) { return move(record.seq); }
  throw out_of_range("End of file reached");
}

}
