/*
Simulating the following 4 files, kmer_size = 3:
  File 1:
    valid
    --empty--
    10 not_found | valid valid
    --empty--
    --empty--
    30 40 50 60 | invalid valid
    --empty--
  File 2 is completely empty:
  File 3:
    --empty--
    --empty--
  File 4:
    70 | valid  valid
    --empty--
    80 invalid 100 | valid valid
*/

#include <climits>
#include <fstream>
#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "BatchObjects/IntervalBatch.h"
#include "BatchObjects/InvalidCharsBatch.h"
#include "BatchObjects/ResultsBatch.h"
#include "ResultsPrinter/ContinuousResultsPrinter.hpp"
#include "Tools/DummyBatchProducer.hpp"
#include "Tools/TestUtils.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::ifstream;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::to_string;
using test_utils::DummyBatchProducer;

using DummyResultsProducer = DummyBatchProducer<ResultsBatch>;
using DummyIntervalProducer = DummyBatchProducer<IntervalBatch>;
using DummyInvalidCharsProducer = DummyBatchProducer<InvalidCharsBatch>;

class DummyContinuousResultsPrinter:
    public ContinuousResultsPrinter<DummyContinuousResultsPrinter> {
  using Base = ContinuousResultsPrinter<DummyContinuousResultsPrinter>;
  friend Base;

private:
  bool is_at_newline = true;
  vector<string> filenames;
  vector<string> result_string;

public:
  [[nodiscard]] auto get_result_string() const -> const vector<string> & {
    return result_string;
  }
  DummyContinuousResultsPrinter(
    const shared_ptr<DummyResultsProducer> &results_producer,
    const shared_ptr<DummyIntervalProducer> &interval_producer,
    const shared_ptr<DummyInvalidCharsProducer> &invalid_chars_producer,
    uint kmer_size,
    uint files
  ):
      filenames(files),
      ContinuousResultsPrinter(
        results_producer,
        interval_producer,
        invalid_chars_producer,
        filenames,
        kmer_size
      ) {}

protected:
  auto do_start_next_file() -> void { result_string.emplace_back(""); }

  auto do_invalid_result() -> void {
    if (!is_at_newline) { result_string.back() += " "; }
    result_string.back() += "INVALID";
    is_at_newline = false;
  }
  auto do_not_found_result() -> void {
    if (!is_at_newline) { result_string.back() += " "; }
    result_string.back() += "NOTFOUND";
    is_at_newline = false;
  }
  auto do_result(size_t result) -> void {
    if (!is_at_newline) { result_string.back() += " "; }
    result_string.back() += to_string(result);
    is_at_newline = false;
  }
  auto do_with_newline() -> void {
    is_at_newline = true;
    result_string.back() += "\n";
  }
};

auto get_results_producer(const vector<ResultsBatch> &results)
  -> shared_ptr<DummyResultsProducer> {
  return make_shared<DummyResultsProducer>(results);
}

auto get_invalid_producer(const vector<InvalidCharsBatch> &invalid_chars)
  -> shared_ptr<DummyInvalidCharsProducer> {
  return make_shared<DummyInvalidCharsProducer>(invalid_chars);
}

auto get_interval_producer(
  const vector<vector<size_t>> &chars_before_newline,
  vector<vector<size_t>> newlines_before_newfile
) -> shared_ptr<DummyIntervalProducer> {
  vector<IntervalBatch> intervals;
  for (int i = 0; i < chars_before_newline.size(); ++i) {
    intervals.push_back({&chars_before_newline[i], newlines_before_newfile[i]});
  }
  return make_shared<DummyIntervalProducer>(intervals);
}

auto get_expected() -> vector<string> {
  return {
    "\n\n10 NOTFOUND\n\n\n30 40 INVALID INVALID\n\n",
    "",
    "\n\n",
    "70\n\nINVALID INVALID 100\n"};
}

class ContinuousResultsPrinterTest: public ::testing::Test {
private:
protected:
  auto run_test(
    uint kmer_size,
    const vector<string> &expected,
    const vector<InvalidCharsBatch> &invalid_chars,
    const vector<ResultsBatch> &results,
    const vector<vector<u64>> &chars_before_newline,
    vector<vector<u64>> newlines_before_newfile,
    uint files
  ) -> void {
    auto results_producer = get_results_producer(results);
    auto invalid_producer = get_invalid_producer(invalid_chars);
    auto interval_producer = get_interval_producer(
      chars_before_newline, std::move(newlines_before_newfile)
    );
    auto results_printer = make_shared<DummyContinuousResultsPrinter>(
      results_producer, interval_producer, invalid_producer, kmer_size, files
    );
    results_printer->read_and_generate();
    ASSERT_EQ(results_printer->get_result_string(), expected);
  }
};

TEST_F(ContinuousResultsPrinterTest, SingleBatch) {
  const uint kmer_size = 3;
  const vector<InvalidCharsBatch> invalid_chars = {{{
    0,
    0,
    0,
    0,
    0,  // end of first string of first file
    0,
    0,
    0,
    0,
    1,
    0,  // end of second string of first file
    0,
    0,
    0,  // end of 4th files first string
    0,
    1,
    0,
    0,
    0  // end of last string
  }}};
  const vector<ResultsBatch> results = {
    {{10, numeric_limits<size_t>::max(), 30, 40, 50, 60, 70, 80, 123456, 100}}};
  const vector<vector<u64>> chars_before_newline = {
    {1, 1, 5, 5, 5, 11, 11, 11, 11, 14, 14, 19, numeric_limits<size_t>::max()}};
  const vector<vector<u64>> newlines_before_newfile
    = {{7, 7, 9, 12, numeric_limits<size_t>::max()}};
  run_test(
    kmer_size,
    get_expected(),
    invalid_chars,
    results,
    chars_before_newline,
    newlines_before_newfile,
    4
  );
}

TEST_F(ContinuousResultsPrinterTest, MultipleBatches) {
  const uint kmer_size = 3;
  const vector<ResultsBatch> results = {
    {{10, numeric_limits<size_t>::max()}},
    {{30, 40, 50, 60, 70}},
    {{80, 123456, 100}}};
  const vector<InvalidCharsBatch> invalid_chars = {
    {{0, 0, 0, 0, 0}},              // end of first string
    {{0, 0, 0, 0, 1, 0, 0, 0, 0}},  // end of third
    {{0, 1, 0, 0, 0}}};             // end of last string
  const vector<vector<u64>> chars_before_newline = {
    {
      1,
      1,
      5,
    },
    {0, 0, 6, 6, 6, 6, 9},
    {0, 5}};
  const vector<vector<u64>> newlines_before_newfile
    = {{ULLONG_MAX}, {4, 4, 6, ULLONG_MAX}, {3, ULLONG_MAX}};
  run_test(
    kmer_size,
    get_expected(),
    invalid_chars,
    results,
    chars_before_newline,
    newlines_before_newfile,
    4
  );
}
}  // namespace sbwt_search
