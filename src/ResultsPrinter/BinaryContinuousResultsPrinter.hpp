#ifndef BINARY_CONTINUOUS_RESULTS_PRINTER_HPP
#define BINARY_CONTINUOUS_RESULTS_PRINTER_HPP

/**
 * @file BinaryContinuousResultsPrinter.hpp
 * @brief Inherits ContinuousResultsPrinter and prints out binary values
 * */

#include "ResultsPrinter/ContinuousResultsPrinter.hpp"
#include "Utils/TypeDefinitions.h"

namespace sbwt_search {

template <
  class ResultsProducer,
  class IntervalProducer,
  class InvalidCharsProducer>
class BinaryContinuousResultsPrinter:
    public ContinuousResultsPrinter<
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer> {
    using Base = ContinuousResultsPrinter<
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer>;

  private:
    u64 minus1 = u64(-1), minus2 = u64(-2), minus3 = u64(-3);

  public:
    BinaryContinuousResultsPrinter(
      shared_ptr<ResultsProducer> results_producer,
      shared_ptr<IntervalProducer> interval_producer,
      shared_ptr<InvalidCharsProducer> invalid_chars_producer,
      vector<string> &filenames,
      uint kmer_size
    ):
        Base(
          results_producer,
          interval_producer,
          invalid_chars_producer,
          filenames,
          kmer_size
        ) {}

  protected:
    auto print_word(
      size_t char_index,
      size_t invalid_index,
      size_t num_chars,
      size_t string_length
    ) -> void override {
      uint invalid_chars_left
        = this->get_invalid_chars_left_first_kmer(invalid_index);
      for (u64 i = char_index; i < char_index + num_chars; ++i) {
        auto furthest_index
          = invalid_index + this->kmer_size - 1 + i - char_index;
        if (furthest_index < invalid_index + string_length && (*this->invalid_chars)[furthest_index]) {
          invalid_chars_left = this->kmer_size;
        }
        if (invalid_chars_left > 0) {
          this->stream.write(reinterpret_cast<char *>(&minus2), sizeof(u64));
          --invalid_chars_left;
        } else if ((*this->results)[i] == u64(-1)) {
          this->stream.write(reinterpret_cast<char *>(&minus1), sizeof(u64));
        } else {
          this->stream.write(
            reinterpret_cast<char *>(&(*this->results)[i]), sizeof(u64)
          );
        }
      }
      this->stream.write(reinterpret_cast<char *>(&minus3), sizeof(u64));
    }
};

}  // namespace sbwt_search

#endif
