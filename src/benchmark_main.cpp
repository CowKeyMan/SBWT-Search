#include <iostream>

#include "BenchmarkUtils.h"
#include "QueryReader.h"

using sbwt_search::QueryReader;
using std::cerr;
using std::cout;

auto QueryReader_parse_kseqpp_streams() -> void;
auto QueryReader_parse_kseqpp_read() -> void;
auto QueryReader_parse_kseqpp_gz_stream() -> void;
auto QueryReader_parse_kseqpp_gz_read() -> void;

const auto fna_filename = "benchmark_objects/GRCh38_latest_genomic.fna";
const auto fna_zipped = "benchmark_objects/GRCh38_latest_genomic.fna.gz";

int main() {
#ifndef BENCHMARK_QUERY_READER
#define BENCHMARK_QUERY_READER
  TIME_IT({QueryReader(fna_filename, 30).parse_kseqpp_streams();})
  cout << "QueryReader_kseqpp_streams: " << TIME_IT_TOTAL << '\n';
  TIME_IT({QueryReader(fna_filename, 30).parse_kseqpp_read();})
  cout << "QueryReader_kseqpp_read: " << TIME_IT_TOTAL << '\n';
  TIME_IT({QueryReader(fna_filename, 30).parse_kseqpp_gz_stream();})
  cout << "QueryReader_kseqpp_gz_stream: " << TIME_IT_TOTAL << '\n';
  TIME_IT({QueryReader(fna_filename, 30).parse_kseqpp_gz_read();})
  cout << "QueryReader_kseqpp_gz_read: " << TIME_IT_TOTAL << '\n';

  TIME_IT({QueryReader(fna_zipped, 30).parse_kseqpp_streams();})
  cout << "QueryReader_kseqpp_streams_zip: " << TIME_IT_TOTAL << '\n';
  TIME_IT({QueryReader(fna_zipped, 30).parse_kseqpp_read();})
  cout << "QueryReader_kseqpp_read_zip: " << TIME_IT_TOTAL << '\n';
  TIME_IT({QueryReader(fna_zipped, 30).parse_kseqpp_gz_stream();})
  cout << "QueryReader_kseqpp_gz_stream_zip: " << TIME_IT_TOTAL << '\n';
  TIME_IT({QueryReader(fna_zipped, 30).parse_kseqpp_gz_read();})
  cout << "QueryReader_kseqpp_gz_read_zip: " << TIME_IT_TOTAL << '\n';
#endif
}
