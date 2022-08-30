#ifndef SHARED_BATCHES_PRODUCER_HPP
#define SHARED_BATCHES_PRODUCER_HPP

/**
 * @file SharedBatchesProducer.hpp
 * @brief Template class for any class which is a continuous batch producer that
 * shares its batch
 * */

#include <memory>
#include <stdexcept>

#include "Utils/CircularBuffer.hpp"
#include "Utils/ErrorUtils.h"
#include "Utils/Semaphore.hpp"

using std::make_shared;
using std::runtime_error;
using std::shared_ptr;
using structure_utils::CircularBuffer;
using threading_utils::Semaphore;

namespace design_utils {

template <class BatchType>
class SharedBatchesProducer {
  private:
    Semaphore start_semaphore, finish_semaphore;
    bool batches_initialised = false;

  protected:
    CircularBuffer<shared_ptr<BatchType>> batches;

    SharedBatchesProducer(const unsigned int max_batches):
        start_semaphore(max_batches - 1),
        finish_semaphore(0),
        batches(max_batches) {}

  public:
    virtual ~SharedBatchesProducer(){};

    auto virtual read_and_generate() -> void {
      throw_if_uninitialised();
      for (unsigned int batch_id = 0; continue_read_condition(); ++batch_id) {
        do_at_batch_start(batch_id);
        generate();
        do_at_batch_finish(batch_id);
      }
      do_at_generate_finish();
    }

    auto operator>>(shared_ptr<BatchType> &out) -> bool {
      throw_if_uninitialised();
      start_semaphore.release();
      finish_semaphore.acquire();
      if (batches.empty()) { return false; }
      out = batches.current_read();
      batches.step_read();
      return true;
    }

  protected:
    auto initialise_batches() -> void {
      for (unsigned int i = 0; i < batches.size(); ++i) {
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
    auto virtual do_at_batch_start(unsigned int batch_id = 0) -> void {
      start_semaphore.acquire();
    }
    auto virtual generate() -> void { throw_uninitialised(); };
    auto virtual do_at_batch_finish(unsigned int batch_id = 0) -> void {
      batches.step_write();
      finish_semaphore.release();
    }
    auto virtual do_at_generate_finish() -> void { finish_semaphore.release(); }

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
