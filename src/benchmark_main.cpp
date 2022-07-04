#include <iostream>
#include <string>

#include <unordered_set>

#include "Utils/BenchmarkUtils.hpp"
#include "QueryFileParser/QueryFileParser.h"
#include "RawSequencesParser/RawSequencesParser.hpp"
#include "cxxopts.hpp"

using std::cerr;
using std::cout;
using std::string;
using std::unordered_set;

using namespace sbwt_search;

auto parse_arguments(int argc, char **argv) -> unordered_set<string>;

auto query_file_parser_fasta() -> void;
auto query_file_parser_fasta_zip() -> void;
auto query_file_parser_fastq() -> void;
auto query_file_parser_fastq_zip() -> void;
u64 kmer_size = 30;
const auto fasta_1gb_path = "benchmark_objects/FASTA1GB.fna";
const auto fasta_1gb_zipped_path = "benchmark_objects/FASTA1GB.fna.gz";
const auto fastq_1gb_path = "benchmark_objects/fastq1GB.fnq";
const auto fastq_1gb_zipped_path = "benchmark_objects/fastq1GB.fnq.gz";

auto raw_sequence_parser_char_to_bits() -> void;

auto main(int argc, char **argv) -> int {
  auto arguments = parse_arguments(argc, argv);
  // QueryFileParser module tests
  if (arguments.count("-QUERY_FILE_PARSER_FASTA")) query_file_parser_fasta();
  if (arguments.count("-QUERY_FILE_PARSER_FASTA_ZIP"))
    query_file_parser_fasta_zip();

  if (arguments.count("-QUERY_FILE_PARSER_FASTQ")) query_file_parser_fastq();
  if (arguments.count("-QUERY_FILE_PARSER_FASTQ_ZIP"))
    query_file_parser_fastq_zip();

  // RawSequencesParser module tests
  if (arguments.count("-RAW_SEQUENCE_PARSER_CHAR_TO_BITS"))
    raw_sequence_parser_char_to_bits();
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

auto query_file_parser_fasta() -> void {
  TIME_IT({ QueryFileParser(fasta_1gb_path, kmer_size).parse_kseqpp_streams(); })
  cout << "QueryFileParser_kseqpp_streams_fasta: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fasta_1gb_path, kmer_size).parse_kseqpp_read(); })
  cout << "QueryFileParser_kseqpp_read_fasta: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fasta_1gb_path, kmer_size).parse_kseqpp_gz_stream(); })
  cout << "QueryFileParser_kseqpp_gz_stream_fasta: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fasta_1gb_path, kmer_size).parse_kseqpp_gz_read(); })
  cout << "QueryFileParser_kseqpp_gz_read_fasta: " << TIME_IT_TOTAL << '\n';
}

auto query_file_parser_fasta_zip() -> void {
  TIME_IT({ QueryFileParser(fasta_1gb_zipped_path, kmer_size).parse_kseqpp_streams(); })
  cout << "QueryFileParser_kseqpp_streams_zip_fasta: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fasta_1gb_zipped_path, kmer_size).parse_kseqpp_read(); })
  cout << "QueryFileParser_kseqpp_read_zip_fasta: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fasta_1gb_zipped_path, kmer_size).parse_kseqpp_gz_stream(); })
  cout << "QueryFileParser_kseqpp_gz_stream_zip_fasta: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fasta_1gb_zipped_path, kmer_size).parse_kseqpp_gz_read(); })
  cout << "QueryFileParser_kseqpp_gz_read_zip_fasta: " << TIME_IT_TOTAL << '\n';
}

auto query_file_parser_fastq() -> void {
  TIME_IT({ QueryFileParser(fastq_1gb_path, kmer_size).parse_kseqpp_streams(); })
  cout << "QueryFileParser_kseqpp_streams_fastq: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fastq_1gb_path, kmer_size).parse_kseqpp_read(); })
  cout << "QueryFileParser_kseqpp_read_fastq: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fastq_1gb_path, kmer_size).parse_kseqpp_gz_stream(); })
  cout << "QueryFileParser_kseqpp_gz_stream_fastq: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fastq_1gb_path, kmer_size).parse_kseqpp_gz_read(); })
  cout << "QueryFileParser_kseqpp_gz_read_fastq: " << TIME_IT_TOTAL << '\n';
}

auto query_file_parser_fastq_zip() -> void {
  TIME_IT({ QueryFileParser(fastq_1gb_zipped_path, kmer_size).parse_kseqpp_streams(); })
  cout << "QueryFileParser_kseqpp_streams_zip_fastq: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fastq_1gb_zipped_path, kmer_size).parse_kseqpp_read(); })
  cout << "QueryFileParser_kseqpp_read_zip_fastq: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fastq_1gb_zipped_path, kmer_size).parse_kseqpp_gz_stream(); })
  cout << "QueryFileParser_kseqpp_gz_stream_zip_fastq: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ QueryFileParser(fastq_1gb_zipped_path, kmer_size).parse_kseqpp_gz_read(); })
  cout << "QueryFileParser_kseqpp_gz_read_zip_fastq: " << TIME_IT_TOTAL << '\n';
}

auto raw_sequence_parser_char_to_bits() -> void {
  auto query_file_parser = QueryFileParser(fasta_1gb_path, kmer_size);
  query_file_parser.parse_kseqpp_streams();
  auto seqs = query_file_parser.get_seqs();
  auto total_positions = query_file_parser.get_total_positions();
  auto total_letters = query_file_parser.get_total_letters();

  TIME_IT({ RawSequencesParser<CharToBitsVector>(seqs, total_positions, total_letters, kmer_size).parse_serial(); })
  cout << "RawSequencesParser_CharToBits_Vector: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ RawSequencesParser<CharToBitsArray>(seqs, total_positions, total_letters, kmer_size).parse_serial(); })
  cout << "RawSequencesParser_CharToBits_Array: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ RawSequencesParser<CharToBitsCArray>(seqs, total_positions, total_letters, kmer_size).parse_serial(); })
  cout << "RawSequencesParser_CharToBits_CArray: " << TIME_IT_TOTAL << '\n';
  TIME_IT({ RawSequencesParser<CharToBitsSwitch>(seqs, total_positions, total_letters, kmer_size).parse_serial(); })
  cout << "RawSequencesParser_CharToBits_Switch: " << TIME_IT_TOTAL << '\n';
}
