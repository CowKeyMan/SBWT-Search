#include <iostream>

#include "BenchmarkUtils.h"
#include "QueryReader.h"

using sbwt_search::QueryReader;
using std::cerr;
using std::cout;

auto QueryReader_parse_kseqpp_streams() -> void;
auto QueryReader_parse_kseqpp_read() -> void;
auto QueryReader_parse_kseqpp_gz_stream() -> void;

const auto fna_filename = "benchmark_objects/GRCh38_latest_genomic.fna";

int main() {
#ifndef BENCHMARK_QUERY_READER
#define BENCHMARK_QUERY_READER

  TIME_IT(QueryReader_parse_kseqpp_streams());
  cout << "QueryReader_read_kseqpp_streams: " << TIME_IT_TOTAL << '\n';

  TIME_IT(QueryReader_parse_kseqpp_read());
  cout << "QueryReader_read_kseqpp_read: " << TIME_IT_TOTAL << '\n';

  TIME_IT(QueryReader_parse_kseqpp_gz_stream());
  cout << "QueryReader_read_gz_stream: " << TIME_IT_TOTAL << '\n';

#endif
}

auto QueryReader_parse_kseqpp_streams() -> void {
  auto query_reader = QueryReader(fna_filename, 30);
  TIME_IT(query_reader.parse_kseqpp_streams());
}

auto QueryReader_parse_kseqpp_read() -> void {
  auto query_reader = QueryReader(fna_filename, 30);
  query_reader.parse_kseqpp_read();
}

auto QueryReader_parse_kseqpp_gz_stream() -> void {
  auto query_reader = QueryReader(fna_filename, 30);
  query_reader.parse_kseqpp_gz_stream();
}
