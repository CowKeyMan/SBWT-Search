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
      BinaryContinuousResultsPrinter<
        ResultsProducer,
        IntervalProducer,
        InvalidCharsProducer>,
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer> {
    using Base = ContinuousResultsPrinter<
      BinaryContinuousResultsPrinter<
        ResultsProducer,
        IntervalProducer,
        InvalidCharsProducer>,
      ResultsProducer,
      IntervalProducer,
      InvalidCharsProducer>;
    friend Base;

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
    auto do_invalid_result() -> void {
      this->stream->write(reinterpret_cast<char *>(&minus2), sizeof(u64));
    }
    auto do_not_found_result() -> void {
      this->stream->write(reinterpret_cast<char *>(&minus1), sizeof(u64));
    }
    auto do_result(size_t result) -> void {
      this->stream->write(reinterpret_cast<char *>(&result), sizeof(u64));
    }
    auto do_with_newline() -> void {
      this->stream->write(reinterpret_cast<char *>(&minus3), sizeof(u64));
    }
};

}  // namespace sbwt_search

#endif
