#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "SequenceFileParser/ContinuousSequenceFileParser.h"
#include "SequenceFileParser/CumulativePropertiesBatchProducer.h"
#include "SequenceFileParser/IntervalBatchProducer.h"
#include "SequenceFileParser/SequenceFileParser.h"
#include "SequenceFileParser/StringSequenceBatchProducer.h"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

namespace sbwt_search {
class CumulativePropertiesBatch;
class IntervalBatch;
class StringSequenceBatch;
}  // namespace sbwt_search

using std::cerr;
using std::runtime_error;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::vector;

namespace sbwt_search {

ContinuousSequenceFileParser::ContinuousSequenceFileParser(
  const vector<string> &filenames,
  const uint kmer_size,
  const u64 max_chars_per_batch,
  const u64 max_strings_per_batch,
  const uint num_readers,
  const u64 max_batches,
  const uint bits_split
):
    filenames(filenames),
    num_readers(num_readers),
    bits_split(bits_split),
    max_strings_per_batch(max_strings_per_batch),
    max_chars_per_batch(get_max_chars_per_batch(max_chars_per_batch, bits_split)
    ),
    string_sequence_batch_producer(
      max_strings_per_batch,
      get_max_chars_per_batch(max_chars_per_batch, bits_split),
      max_batches,
      num_readers,
      bits_split
    ),
    cumulative_properties_batch_producer(
      max_batches, max_strings_per_batch, kmer_size
    ),
    interval_batch_producer(max_batches, max_strings_per_batch) {}

auto ContinuousSequenceFileParser::read_and_generate() -> void {
  start_new_batch();
  for (auto &filename: filenames) {
    try {
      process_file(filename);
    } catch (runtime_error &e) { cerr << e.what() << '\n'; }
    interval_batch_producer.file_end();
  }
  terminate_batch();
  string_sequence_batch_producer.set_finished_reading();
  cumulative_properties_batch_producer.set_finished_reading();
  interval_batch_producer.set_finished_reading();
}

auto ContinuousSequenceFileParser::operator>>(
  shared_ptr<StringSequenceBatch> &batch
) -> bool {
  return string_sequence_batch_producer >> batch;
}

auto ContinuousSequenceFileParser::operator>>(
  shared_ptr<CumulativePropertiesBatch> &batch
) -> bool {
  return cumulative_properties_batch_producer >> batch;
}

auto ContinuousSequenceFileParser::operator>>(shared_ptr<IntervalBatch> &batch)
  -> bool {
  return interval_batch_producer >> batch;
}

auto ContinuousSequenceFileParser::get_max_chars_per_batch(
  u64 value, uint bits_split
) -> u64 {
  auto result = round_down<u64>(value, bits_split / 2);
  if (result == 0) { result = bits_split / 2; };
  return result;
}

auto ContinuousSequenceFileParser::start_new_batch() -> void {
  string_sequence_batch_producer.start_new_batch();
  cumulative_properties_batch_producer.start_new_batch();
  interval_batch_producer.start_new_batch();
  current_batch_size = current_batch_strings = 0;
}

auto ContinuousSequenceFileParser::terminate_batch() -> void {
  string_sequence_batch_producer.terminate_batch();
  cumulative_properties_batch_producer.terminate_batch();
  interval_batch_producer.terminate_batch();
}

auto ContinuousSequenceFileParser::process_file(const string &filename)
  -> void {
  SequenceFileParser parser(filename);
  string s;
  u64 string_index = 0;
  while (parser >> s) { process_string(filename, s, string_index++); }
}

auto ContinuousSequenceFileParser::process_string(
  const string &filename, string &s, const u64 string_index
) -> void {
  if (string_larger_than_limit(s)) {
    print_string_too_large(filename, string_index);
    interval_batch_producer.add_string("");
    return;
  }
  if (!string_fits_in_batch(s) || current_batch_strings > max_strings_per_batch) {
    terminate_batch();
    start_new_batch();
  }
  add_string(s);
  ++current_batch_strings;
}

auto ContinuousSequenceFileParser::add_string(string &s) -> void {
  string_sequence_batch_producer.add_string(s);
  cumulative_properties_batch_producer.add_string(s);
  interval_batch_producer.add_string(s);
  current_batch_size += s.size();
}

auto ContinuousSequenceFileParser::string_fits_in_batch(const string &s)
  -> bool {
  return s.size() + current_batch_size <= max_chars_per_batch;
}

auto ContinuousSequenceFileParser::string_larger_than_limit(const string &s)
  -> bool {
  return s.size() > max_chars_per_batch;
}

auto ContinuousSequenceFileParser::print_string_too_large(
  const string &filename, const uint string_index
) -> void {
  cerr << "The string in file " + filename + " at position "
            + to_string(string_index) + " is too large\n";
}

}  // namespace sbwt_search
