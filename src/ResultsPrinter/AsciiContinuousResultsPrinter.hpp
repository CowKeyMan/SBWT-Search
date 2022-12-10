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
      AsciiContinuousResultsPrinter<
        ResultsProducer,
        IntervalProducer,
        InvalidCharsProducer>,
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer> {
    using Base = ContinuousResultsPrinter<
      AsciiContinuousResultsPrinter<
        ResultsProducer,
        IntervalProducer,
        InvalidCharsProducer>,
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer>;
    friend Base;

  private:
    bool is_at_newline = true;

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
    auto do_invalid_result() -> void {
      if (!is_at_newline) { (*this->stream) << " "; }
      (*this->stream) << "-2";
      is_at_newline = false;
    }
    auto do_not_found_result() -> void {
      if (!is_at_newline) { (*this->stream) << " "; }
      (*this->stream) << "-1";
      is_at_newline = false;
    }
    auto do_result(size_t result) -> void {
      if (!is_at_newline) { (*this->stream) << " "; }
      (*this->stream) << result;
      is_at_newline = false;
    }
    auto do_with_newline() -> void {
      (*this->stream) << "\n";
      is_at_newline = true;
    }
};

}  // namespace sbwt_search

#endif
