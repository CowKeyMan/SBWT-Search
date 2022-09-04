#ifndef ASCII_CONTINUOUS_RESULTS_PRINTER_HPP
#define ASCII_CONTINUOUS_RESULTS_PRINTER_HPP

/**
 * @file AsciiContinuousResultsPrinter.hpp
 * @brief Inherits ContinuousResultsPrinter and prints out ascii values
 * */

#include "ResultsPrinter/ContinuousResultsPrinter.hpp"

namespace sbwt_search {

template <
  class ResultsProducer,
  class IntervalProducer,
  class InvalidCharsProducer>
class AsciiContinuousResultsPrinter:
    public ContinuousResultsPrinter<
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer> {
    using Base = ContinuousResultsPrinter<
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer>;

  public:
    AsciiContinuousResultsPrinter(
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
          this->stream << "-2";
          --invalid_chars_left;
        } else if ((*this->results)[i] == u64(-1)) {
          this->stream << "-1";
        } else {
          this->stream << (*this->results)[i];
        }
        if (i + 1 != char_index + num_chars) { this->stream << ' '; }
      }
      this->stream << '\n';
    }
};

}  // namespace sbwt_search

#endif
