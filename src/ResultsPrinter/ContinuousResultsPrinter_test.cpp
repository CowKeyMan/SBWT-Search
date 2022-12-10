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
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "BatchObjects/IntervalBatch.h"
#include "BatchObjects/InvalidCharsBatch.h"
#include "BatchObjects/ResultsBatch.h"
#include "ResultsPrinter/ContinuousResultsPrinter.hpp"
#include "TestUtils/GeneralTestUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::getline;
using std::ifstream;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::filesystem::remove;

namespace sbwt_search {

using DummyResultsProducer = DummyBatchProducer<ResultsBatch>;
using DummyIntervalProducer = DummyBatchProducer<IntervalBatch>;
using DummyInvalidCharsProducer = DummyBatchProducer<InvalidCharsBatch>;

class DummyContinuousResultsPrinter:
    public ContinuousResultsPrinter<
      DummyContinuousResultsPrinter,
      DummyResultsProducer,
      DummyIntervalProducer,
      DummyInvalidCharsProducer> {
    using Base = ContinuousResultsPrinter<
      DummyContinuousResultsPrinter,
      DummyResultsProducer,
      DummyIntervalProducer,
      DummyInvalidCharsProducer>;
    friend Base;

  private:
    bool is_at_newline = true;
    vector<string> filenames;

  public:
    vector<string> result_string;
    DummyContinuousResultsPrinter(
      shared_ptr<DummyResultsProducer> results_producer,
      shared_ptr<DummyIntervalProducer> interval_producer,
      shared_ptr<DummyInvalidCharsProducer> invalid_chars_producer,
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
    auto do_start_next_file() -> void { result_string.push_back(""); }

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

auto get_results_producer(vector<ResultsBatch> results)
  -> shared_ptr<DummyResultsProducer> {
  return make_shared<DummyResultsProducer>(results);
}

auto get_invalid_producer(vector<InvalidCharsBatch> invalid_chars)
  -> shared_ptr<DummyInvalidCharsProducer> {
  return make_shared<DummyInvalidCharsProducer>(invalid_chars);
}

auto get_interval_producer(
  const vector<vector<size_t>> &chars_before_newline,
  vector<vector<size_t>> newlines_before_newfile
) -> shared_ptr<DummyIntervalProducer> {
  vector<IntervalBatch> intervals;
  for (int i = 0; i < chars_before_newline.size(); ++i) {
    intervals.push_back({ &chars_before_newline[i], newlines_before_newfile[i] }
    );
  }
  return make_shared<DummyIntervalProducer>(intervals);
}

class ContinuousResultsPrinterTest: public ::testing::Test {
  protected:
    auto run_test(
      uint kmer_size,
      vector<string> expected,
      vector<InvalidCharsBatch> invalid_chars,
      vector<ResultsBatch> results,
      vector<vector<u64>> chars_before_newline,
      vector<vector<u64>> newlines_before_newfile,
      uint files
    ) -> void {
      auto results_producer = get_results_producer(results);
      auto invalid_producer = get_invalid_producer(invalid_chars);
      auto interval_producer
        = get_interval_producer(chars_before_newline, newlines_before_newfile);
      auto results_printer = make_shared<DummyContinuousResultsPrinter>(
        results_producer, interval_producer, invalid_producer, kmer_size, files
      );
      results_printer->read_and_generate();
      ASSERT_EQ(results_printer->result_string, expected);
    }
};

vector<string> expected = { "\n\n10 NOTFOUND\n\n\n30 40 INVALID INVALID\n\n",
                            "",
                            "\n\n",
                            "70\n\nINVALID INVALID 100\n" };

TEST_F(ContinuousResultsPrinterTest, SingleBatch) {
  const uint kmer_size = 3;
  vector<InvalidCharsBatch> invalid_chars = { { {
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
  } } };
  vector<ResultsBatch> results
    = { { { 10, size_t(-1), 30, 40, 50, 60, 70, 80, 123456, 100 } } };
  vector<vector<u64>> chars_before_newline
    = { { 1, 1, 5, 5, 5, 11, 11, 11, 11, 14, 14, 19, size_t(-1) } };
  vector<vector<u64>> newlines_before_newfile = { { 7, 7, 9, 12, size_t(-1) } };
  run_test(
    kmer_size,
    expected,
    invalid_chars,
    results,
    chars_before_newline,
    newlines_before_newfile,
    4
  );
}

TEST_F(ContinuousResultsPrinterTest, MultipleBatches) {
  const uint kmer_size = 3;
  vector<ResultsBatch> results = { { { 10, size_t(-1) } },
                                   { { 30, 40, 50, 60, 70 } },
                                   { { 80, 123456, 100 } } };
  vector<InvalidCharsBatch> invalid_chars
    = { { { 0, 0, 0, 0, 0 } },  // end of first string
        { { 0, 0, 0, 0, 1, 0, 0, 0, 0 } },  // end of third
        { { 0, 1, 0, 0, 0 } } };  // end of last string
  vector<vector<u64>> chars_before_newline = { {
                                                 1,
                                                 1,
                                                 5,
                                               },
                                               { 0, 0, 6, 6, 6, 6, 9 },
                                               { 0, 5 } };
  vector<vector<u64>> newlines_before_newfile
    = { { ULLONG_MAX }, { 4, 4, 6, ULLONG_MAX }, { 3, ULLONG_MAX } };
  run_test(
    kmer_size,
    expected,
    invalid_chars,
    results,
    chars_before_newline,
    newlines_before_newfile,
    4
  );
}
}  // namespace sbwt_search
