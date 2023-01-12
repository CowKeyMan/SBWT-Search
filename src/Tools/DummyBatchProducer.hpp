#ifndef DUMMY_BATCH_PRODUCER_HPP
#define DUMMY_BATCH_PRODUCER_HPP

/**
 * @file DummyBatchProducer.hpp
 * @brief Inheriting from the SharedBatchesProducer, this class sequentially
 * produces the batches it was given in its constructor. Useful for testing
 */

#include <memory>

#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"

namespace test_utils {

using design_utils::SharedBatchesProducer;
using std::make_shared;
using std::shared_ptr;

template <class T>
class DummyBatchProducer: public SharedBatchesProducer<T> {
  vector<shared_ptr<T>> batches;
  uint counter = 0;

public:
  explicit DummyBatchProducer(const vector<shared_ptr<T>> &batches_):
    batches(batches_), SharedBatchesProducer<T>(batches_.size() + 1) {
    this->initialise_batches();
    this->read_and_generate();
  }
  explicit DummyBatchProducer(const vector<T> &batches_):
    SharedBatchesProducer<T>(batches_.size() + 1) {
    this->initialise_batches();
    for (auto b : batches_) { batches.push_back(make_shared<T>(b)); }
    this->read_and_generate();
  }

  auto get_default_value() -> shared_ptr<T> override {
    return make_shared<T>();
  }

  auto continue_read_condition() -> bool override {
    return !(counter == batches.size());
  }
  auto generate() -> void override {
    this->get_batches().current_write() = batches[counter];
    counter++;
  }
};

}  // namespace test_utils

#endif
