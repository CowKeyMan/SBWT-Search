#include <iostream>

#include "BenchmarkUtils.h"
#include "QueryReader.h"

using sbwt_search::QueryReader;
using std::cerr;
using std::cout;

auto QueryReader_parse_kseqpp_streams() -> void;
auto QueryReader_parse_kseqpp_read() -> void;

int main() {
#ifndef BENCHMARK_QUERY_READER
#define BENCHMARK_QUERY_READER

  TIME_IT(QueryReader_parse_kseqpp_streams());
  cout << "QueryReader_read_kseqpp_streams: " << TIME_IT_TOTAL << '\n';

  TIME_IT(QueryReader_parse_kseqpp_read());
  cout << "QueryReader_read_kseqpp_read: " << TIME_IT_TOTAL << '\n';

#endif
}

auto QueryReader_parse_kseqpp_streams() -> void {
  auto query_reader = QueryReader("data/QueryData/365.fna", 30);
  TIME_IT(query_reader.parse_kseqpp_streams());
}

auto QueryReader_parse_kseqpp_read() -> void {
  auto query_reader = QueryReader("data/QueryData/365.fna", 30);
  TIME_IT(query_reader.parse_kseqpp_read());
}
