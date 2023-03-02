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
#include "ResultsPrinter/BoolContinuousResultsPrinter.h"
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
  shared_ptr<BinaryContinuousResultsPrinter>,
  shared_ptr<BoolContinuousResultsPrinter>>;

class IndexSearchMain: public Main {
public:
  auto main(int argc, char **argv) -> int override;

private:
  u64 kmer_size = 0;
  u64 max_chars_per_batch = 0;
  unique_ptr<IndexSearchArgumentParser> args;

  [[nodiscard]] auto get_args() const -> const IndexSearchArgumentParser &;
  auto get_gpu_container() -> shared_ptr<GpuSbwtContainer>;
  auto load_batch_info() -> void;
  auto get_max_chars_per_batch_cpu() -> u64;
  auto get_max_chars_per_batch_gpu() -> u64;
  auto get_max_chars_per_batch() -> u64;
  auto get_components(
    const shared_ptr<GpuSbwtContainer> &gpu_container,
    const vector<string> &input_filenames,
    const vector<string> &output_filenames
  )
    -> tuple<
      shared_ptr<ContinuousSequenceFileParser>,
      shared_ptr<ContinuousSeqToBitsConverter>,
      shared_ptr<ContinuousPositionsBuilder>,
      shared_ptr<ContinuousSearcher>,
      ResultsPrinter>;
  auto load_input_output_filenames(
    const string &input_file, const string &output_file
  ) -> void;
  auto get_input_output_filenames()
    -> tuple<vector<vector<string>>, vector<vector<string>>>;
  auto get_results_printer(
    const shared_ptr<ContinuousSearcher> &searcher,
    const shared_ptr<IntervalBatchProducer> &interval_batch_producer,
    const shared_ptr<InvalidCharsProducer> &invalid_chars_producer,
    const vector<string> &output_filenames
  ) -> ResultsPrinter;
  auto
  run_components(shared_ptr<ContinuousSequenceFileParser> &, shared_ptr<ContinuousSeqToBitsConverter> &, shared_ptr<ContinuousPositionsBuilder> &, shared_ptr<ContinuousSearcher> &, ResultsPrinter &)
    -> void;
};

}  // namespace sbwt_search

#endif
