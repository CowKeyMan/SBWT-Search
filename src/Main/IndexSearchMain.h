#ifndef INDEX_SEARCH_MAIN_H
#define INDEX_SEARCH_MAIN_H

/**
 * @file IndexSearchMain.h
 * @brief The main function for searching the index. The 'index' mode of the
 * main executable.
 */

#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "ArgumentParser/IndexSearchArgumentParser.h"
#include "Main/Main.h"
#include "PositionsBuilder/ContinuousPositionsBuilder.h"
#include "ResultsPrinter/AsciiContinuousResultsPrinter.h"
#include "ResultsPrinter/BinaryContinuousResultsPrinter.h"
#include "SbwtContainer/GpuSbwtContainer.h"
#include "Searcher/ContinuousSearcher.h"
#include "SeqToBitsConverter/ContinuousSeqToBitsConverter.h"
#include "SequenceFileParser/ContinuousSequenceFileParser.h"

namespace sbwt_search {

using std::shared_ptr;
using std::string;
using std::tuple;
using std::variant;
using std::vector;

using ResultsPrinter = variant<
  shared_ptr<AsciiContinuousResultsPrinter>,
  shared_ptr<BinaryContinuousResultsPrinter>>;

class IndexSearchMain: public Main {
public:
  auto main(int argc, char **argv) -> int override;

private:
  u64 kmer_size = 0;
  u64 streams = 0;
  u64 max_chars_per_batch = 0;
  u64 max_reads_per_batch = 0;
  unique_ptr<IndexSearchArgumentParser> args;

  [[nodiscard]] auto get_args() const -> const IndexSearchArgumentParser &;
  auto get_gpu_container() -> shared_ptr<GpuSbwtContainer>;
  auto load_batch_info() -> void;
  auto get_max_chars_per_batch_cpu() -> u64;
  auto get_max_chars_per_batch_gpu() -> u64;
  auto get_max_chars_per_batch() -> u64;
  auto get_components(
    const shared_ptr<GpuSbwtContainer> &gpu_container,
    const vector<vector<string>> &input_filenames,
    const vector<vector<string>> &output_filenames
  )
    -> tuple<
      vector<shared_ptr<ContinuousSequenceFileParser>>,
      vector<shared_ptr<ContinuousSeqToBitsConverter>>,
      vector<shared_ptr<ContinuousPositionsBuilder>>,
      vector<shared_ptr<ContinuousSearcher>>,
      vector<ResultsPrinter>>;
  auto load_input_output_filenames(
    const string &input_file, const string &output_file
  ) -> void;
  auto get_input_output_filenames()
    -> tuple<vector<vector<string>>, vector<vector<string>>>;
  auto get_results_printer(
    const shared_ptr<ContinuousSearcher> &searcher,
    const shared_ptr<IntervalBatchProducer> &interval_batch_producer,
    const shared_ptr<InvalidCharsProducer> &invalid_chars_producer,
    const vector<string> &split_output_filenames
  ) -> ResultsPrinter;
  auto run_components(
    vector<shared_ptr<ContinuousSequenceFileParser>> &sequence_file_parsers,
    vector<shared_ptr<ContinuousSeqToBitsConverter>> &seq_to_bits_converters,
    vector<shared_ptr<ContinuousPositionsBuilder>> &positions_builders,
    vector<shared_ptr<ContinuousSearcher>> &searchers,
    vector<ResultsPrinter> &results_printers
  ) -> void;
};

}  // namespace sbwt_search

#endif
