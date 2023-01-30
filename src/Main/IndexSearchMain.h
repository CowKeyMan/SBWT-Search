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
  vector<string> input_filenames;
  vector<string> output_filenames;
  u64 kmer_size = 0;
  u64 max_chars_per_batch = 0;
  u64 max_batches = 0;
  u64 threads = 0;

private:
  auto get_gpu_container(const string &index_file)
    -> shared_ptr<GpuSbwtContainer>;
  auto load_input_output_filenames(
    const string &input_file, const string &output_file
  ) -> void;
  auto load_batch_info(u64 max_batches, u64 unavailable_ram, u64 max_cpu_memory)
    -> void;
  auto get_max_chars_per_batch_cpu(size_t unavailable_memory, size_t max_memory)
    -> size_t;
  auto get_max_chars_per_batch_gpu() -> size_t;
  auto get_max_chars_per_batch(size_t unavailable_memory, size_t max_cpu_memory)
    -> size_t;
  auto load_threads() -> void;
  auto get_components(
    shared_ptr<GpuSbwtContainer> gpu_container, const string &print_mode
  )
    -> std::tuple<
      shared_ptr<ContinuousSequenceFileParser>,
      shared_ptr<ContinuousSeqToBitsConverter>,
      shared_ptr<ContinuousPositionsBuilder>,
      shared_ptr<ContinuousSearcher>,
      ResultsPrinter>;
  auto get_results_printer(
    const string &print_mode,
    const shared_ptr<ContinuousSearcher> &searcher,
    const shared_ptr<IntervalBatchProducer> &interval_batch_producer,
    const shared_ptr<InvalidCharsProducer> &invalid_chars_producer
  ) -> ResultsPrinter;
  auto
  run_components(shared_ptr<ContinuousSequenceFileParser> &, shared_ptr<ContinuousSeqToBitsConverter> &, shared_ptr<ContinuousPositionsBuilder> &, shared_ptr<ContinuousSearcher> &, ResultsPrinter &)
    -> void;
};

}  // namespace sbwt_search

#endif
