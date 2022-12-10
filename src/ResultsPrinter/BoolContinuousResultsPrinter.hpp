#ifndef BOOL_CONTINUOUS_RESULTS_PRINTER_HPP
#define BOOL_CONTINUOUS_RESULTS_PRINTER_HPP

/**
 * @file BoolContinuousResultsPrinter.hpp
 * @brief Inherits ContinuousResultsPrinter and prints out boolean bitvectors
 * */

#include <algorithm>
#include <memory>

#include "ResultsPrinter/ContinuousResultsPrinter.hpp"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using math_utils::round_up;
using std::fill;
using std::ios_base;
using std::unique_ptr;

namespace sbwt_search {

template <
  class ResultsProducer,
  class IntervalProducer,
  class InvalidCharsProducer>
class BoolContinuousResultsPrinter:
    public ContinuousResultsPrinter<
      BoolContinuousResultsPrinter<
        ResultsProducer,
        IntervalProducer,
        InvalidCharsProducer>,
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer> {
    using Base = ContinuousResultsPrinter<
      BoolContinuousResultsPrinter<
        ResultsProducer,
        IntervalProducer,
        InvalidCharsProducer>,
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer>;
    friend Base;

  private:
    vector<char> batch;
    u64 working_bits = 0, working_seq_size = 0;
    u8 shift_bits = 63;
    unique_ptr<ThrowingOfstream> seq_size_stream;

  public:
    BoolContinuousResultsPrinter(
      shared_ptr<ResultsProducer> results_producer,
      shared_ptr<IntervalProducer> interval_producer,
      shared_ptr<InvalidCharsProducer> invalid_chars_producer,
      vector<string> &filenames,
      uint kmer_size,
      u64 max_chars_per_batch
    ):
        Base(
          results_producer,
          interval_producer,
          invalid_chars_producer,
          filenames,
          kmer_size
        ),
        batch(round_up<u64>(max_chars_per_batch, 8) / 8) {
      reset_working_bits();
    }

  protected:
    auto do_invalid_result() -> void { shift(); }

    auto do_not_found_result() -> void { shift(); }

    auto do_result(size_t result) -> void {
      working_bits |= (1ULL << shift_bits);
      shift();
    }

    auto do_with_newline() -> void {
      this->seq_size_stream->write(
        reinterpret_cast<char *>(&working_seq_size), sizeof(u64)
      );
      working_seq_size = 0;
    }

    auto do_at_file_end() -> void { dump_working_bits(); }

    auto do_start_next_file() -> void {
      seq_size_stream = make_unique<ThrowingOfstream>(
        (*this->current_filename) + "_seq_sizes",
        ios_base::binary | ios_base::out
      );
      Base::do_start_next_file();
    }

  private:
    auto reset_working_bits() -> void {
      working_bits = 0;
      shift_bits = 63;
    }

    auto shift() -> void {
      ++working_seq_size;
      if (shift_bits == 0) {
        dump_working_bits();
        return;
      }
      --shift_bits;
    }

    auto dump_working_bits() -> void {
      this->stream->write(reinterpret_cast<char *>(&working_bits), sizeof(u64));
      reset_working_bits();
    }
};

}  // namespace sbwt_search

#endif
