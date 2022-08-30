#ifndef CONTINUOUS_SEQUENCE_FILE_PARSER_H
#define CONTINUOUS_SEQUENCE_FILE_PARSER_H

/**
 * @file ContinuousSequenceFileParser.h
 * @brief Continuously reads sequences from a file or multiple files into a
 * buffer. Then it can serve these sequences to its consumers
 * */

#include <memory>
#include <string>
#include <vector>

#include "SequenceFileParser/CumulativePropertiesBatchProducer.h"
#include "SequenceFileParser/IntervalBatchProducer.h"
#include "SequenceFileParser/StringSequenceBatchProducer.h"
#include "Utils/TypeDefinitions.h"

namespace sbwt_search {
class CumulativePropertiesBatch;
class IntervalBatch;
class StringSequenceBatch;
}  // namespace sbwt_search

using std::shared_ptr;
using std::string;
using std::vector;

namespace sbwt_search {

class ContinuousSequenceFileParser {
  private:
    const u64 max_chars_per_batch, max_strings_per_batch;
    u64 current_batch_size = 0, current_batch_strings = 0;
    const uint num_readers;
    const vector<string> filenames;
    StringSequenceBatchProducer string_sequence_batch_producer;
    CumulativePropertiesBatchProducer cumulative_properties_batch_producer;
    IntervalBatchProducer interval_batch_producer;
    const uint bits_split = 64;
    uint batch_idx = 0;

  public:
    ContinuousSequenceFileParser(
      const vector<string> &filenames,
      const uint kmer_size,
      const u64 max_chars_per_batch = 1000,
      const u64 max_strings_per_batch = 1000,
      const uint num_readers = 1,
      const u64 max_batches = 5,
      const uint bits_split = 64
    );

    void read_and_generate();
    bool operator>>(shared_ptr<StringSequenceBatch> &batch);
    bool operator>>(shared_ptr<CumulativePropertiesBatch> &batch);
    bool operator>>(shared_ptr<IntervalBatch> &batch);

  private:
    u64 get_max_chars_per_batch(u64 value, uint bits_split);
    void start_new_batch();
    void terminate_batch();
    void process_file(const string &filename);
    void
    process_string(const string &filename, string &s, const u64 string_index);
    void add_string(string &s);
    bool string_fits_in_batch(const string &s);
    bool string_larger_than_limit(const string &s);
    void print_string_too_large(
      const string &filename, const u64 string_index, const u64 string_size
    );
};

}  // namespace sbwt_search
#endif
