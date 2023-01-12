#ifndef SHARED_BATCHES_PRODUCER_HPP
#define SHARED_BATCHES_PRODUCER_HPP

/**
 * @file SharedBatchesProducer.hpp
 * @brief Template class for any class which is a continuous batch producer that
 * shares its batch
 */

#include <memory>
#include <stdexcept>

#include "Tools/CircularBuffer.hpp"
#include "Tools/ErrorUtils.h"
#include "Tools/OmpLock.h"
#include "Tools/Semaphore.h"

namespace design_utils {

using std::make_shared;
using std::runtime_error;
using std::shared_ptr;
using structure_utils::CircularBuffer;
using threading_utils::OmpLock;
using threading_utils::Semaphore;

template <class BatchType>
class SharedBatchesProducer {
private:
  Semaphore start_semaphore, finish_semaphore;
  bool batches_initialised = false;
  OmpLock step_lock;
  uint batch_id = 0;
  CircularBuffer<shared_ptr<BatchType>> batches;

public:
  SharedBatchesProducer(SharedBatchesProducer &) = delete;
  SharedBatchesProducer(SharedBatchesProducer &&) = delete;
  auto operator=(SharedBatchesProducer &) = delete;
  auto operator=(SharedBatchesProducer &&) = delete;

  explicit SharedBatchesProducer(const unsigned int max_batches):
    // the '-1' is so that we do not overwrite the current item being read
    start_semaphore(max_batches - 1),
    finish_semaphore(0),
    batches(max_batches) {}

  auto virtual read_and_generate() -> void {
    throw_if_uninitialised();
    for (batch_id = 0; continue_read_condition(); ++batch_id) {
      do_at_batch_start();
      generate();
      do_at_batch_finish();
    }
    do_at_generate_finish();
  }

  auto operator>>(shared_ptr<BatchType> &out) -> bool {
    start_semaphore.release();
    finish_semaphore.acquire();
    step_lock.set_lock();
    if (batches.empty()) {
      step_lock.unset_lock();
      return false;
    }
    out = batches.current_read();
    batches.step_read();
    step_lock.unset_lock();
    return true;
  }

protected:
  [[nodiscard]] auto get_batch_id() const -> uint { return batch_id; }
  [[nodiscard]] auto get_batches() -> CircularBuffer<shared_ptr<BatchType>> & {
    return batches;
  }

  auto initialise_batches() -> void {
    for (unsigned int i = 0; i < batches.capacity(); ++i) {
      batches.set(i, get_default_value());
    }
    batches_initialised = true;
  }
  auto virtual get_default_value() -> shared_ptr<BatchType> {
    throw_uninitialised();
    return nullptr;
  };
  auto virtual continue_read_condition() -> bool {
    throw_uninitialised();
    return false;
  };
  auto virtual do_at_batch_start() -> void { start_semaphore.acquire(); }
  auto virtual generate() -> void { throw_uninitialised(); };
  auto virtual do_at_batch_finish() -> void {
    step_lock.set_lock();
    batches.step_write();
    step_lock.unset_lock();
    finish_semaphore.release();
  }
  auto virtual do_at_generate_finish() -> void { finish_semaphore.release(); }
  virtual ~SharedBatchesProducer() = default;

private:
  auto throw_if_uninitialised() {
    if (!batches_initialised) {
      throw runtime_error(
        "batches not initialised. Please call initialise_batches() first."
      );
    }
  }
};

}  // namespace design_utils
#endif
