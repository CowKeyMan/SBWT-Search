#include <iostream>
#include <string>

#include <unordered_set>

#include "BenchmarkUtils.h"
#include "QueryFileParser.h"
#include "cxxopts.hpp"

using sbwt_search::QueryFileParser;
using std::cerr;
using std::cout;
using std::string;
using std::unordered_set;

auto parse_arguments(int argc, char **argv) -> unordered_set<string>;

auto QueryFileParser_parse_kseqpp_streams() -> void;
auto QueryFileParser_parse_kseqpp_read() -> void;
auto QueryFileParser_parse_kseqpp_gz_stream() -> void;
auto QueryFileParser_parse_kseqpp_gz_read() -> void;

const auto fna_filename = "benchmark_objects/GRCh38_latest_genomic.fna";
const auto fna_zipped = "benchmark_objects/GRCh38_latest_genomic.fna.gz";

int main(int argc, char **argv) {
  auto arguments = parse_arguments(argc, argv);

  if (arguments.count("-QUERY_FILE_PARSER_FASTA")) {
    TIME_IT({ QueryFileParser(fna_filename, 30).parse_kseqpp_streams(); })
    cout << "QueryFileParser_kseqpp_streams: " << TIME_IT_TOTAL << '\n';
    TIME_IT({ QueryFileParser(fna_filename, 30).parse_kseqpp_read(); })
    cout << "QueryFileParser_kseqpp_read: " << TIME_IT_TOTAL << '\n';
    TIME_IT({ QueryFileParser(fna_filename, 30).parse_kseqpp_gz_stream(); })
    cout << "QueryFileParser_kseqpp_gz_stream: " << TIME_IT_TOTAL << '\n';
    TIME_IT({ QueryFileParser(fna_filename, 30).parse_kseqpp_gz_read(); })
    cout << "QueryFileParser_kseqpp_gz_read: " << TIME_IT_TOTAL << '\n';
  }

  if (arguments.count("-QUERY_FILE_PARSER_FASTA_ZIP")) {
    TIME_IT({ QueryFileParser(fna_zipped, 30).parse_kseqpp_streams(); })
    cout << "QueryFileParser_kseqpp_streams_zip: " << TIME_IT_TOTAL << '\n';
    TIME_IT({ QueryFileParser(fna_zipped, 30).parse_kseqpp_read(); })
    cout << "QueryFileParser_kseqpp_read_zip: " << TIME_IT_TOTAL << '\n';
    TIME_IT({ QueryFileParser(fna_zipped, 30).parse_kseqpp_gz_stream(); })
    cout << "QueryFileParser_kseqpp_gz_stream_zip: " << TIME_IT_TOTAL << '\n';
    TIME_IT({ QueryFileParser(fna_zipped, 30).parse_kseqpp_gz_read(); })
    cout << "QueryFileParser_kseqpp_gz_read_zip: " << TIME_IT_TOTAL << '\n';
  }
}

auto parse_arguments(int argc, char **argv) -> unordered_set<string> {
  auto options = cxxopts::Options("Benchmark");
  options.allow_unrecognised_options();
  auto local_arguments = options.parse(argc, argv);
  auto unmatched_args = local_arguments.unmatched();
  return unordered_set<string>(
    std::make_move_iterator(unmatched_args.begin()),
    std::make_move_iterator(unmatched_args.end())
  );
}
