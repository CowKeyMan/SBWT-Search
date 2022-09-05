#ifndef BOOL_CONTINUOUS_RESULTS_PRINTER_HPP
#define BOOL_CONTINUOUS_RESULTS_PRINTER_HPP

/**
 * @file BoolContinuousResultsPrinter.hpp
 * @brief Inherits ContinuousResultsPrinter and prints out boolean bitvectors
 * */

#include <algorithm>

#include "ResultsPrinter/ContinuousResultsPrinter.hpp"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using math_utils::round_up;
using std::fill;

namespace sbwt_search {

template <
  class ResultsProducer,
  class IntervalProducer,
  class InvalidCharsProducer>
class BoolContinuousResultsPrinter:
    public ContinuousResultsPrinter<
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer> {
    using Base = ContinuousResultsPrinter<
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer>;

  private:
    vector<char> batch;

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
        batch(round_up<u64>(max_chars_per_batch, 8) / 8) {}

  protected:
    auto print_word(
      size_t char_index,
      size_t invalid_index,
      size_t num_chars,
      size_t string_length
    ) -> void override {
      auto to_write = round_up<u64>(num_chars, 8) / 8;
      fill(batch.begin(), batch.begin() + to_write, 0);
      uint invalid_chars_left
        = this->get_invalid_chars_left_first_kmer(invalid_index);
      for (u64 i = char_index; i < char_index + num_chars; ++i) {
        auto furthest_index
          = invalid_index + this->kmer_size - 1 + i - char_index;
        if (furthest_index < invalid_index + string_length && (*this->invalid_chars)[furthest_index]) {
          invalid_chars_left = this->kmer_size;
        }
        if (invalid_chars_left > 0) {
          --invalid_chars_left;
        } else if ((*this->results)[i] != u64(-1)) {
          auto batch_index = (i - char_index);
          this->batch[batch_index / 8] |= (1 << (7 - batch_index % 8));
        }
      }
      this->stream.write(
        reinterpret_cast<char *>(&num_chars), sizeof(num_chars)
      );
      this->stream.write(&batch[0], to_write);
    }
};

}  // namespace sbwt_search

#endif
