#ifndef SHARED_BATCHES_PRODUCER_HPP
#define SHARED_BATCHES_PRODUCER_HPP

/**
 * @file SharedBatchesProducer.hpp
 * @brief Template class for any class which is a continuous batch producer that
 * shares its batch
 * */

#include <memory>

#include "Utils/CircularBuffer.hpp"
#include "Utils/Semaphore.hpp"

using std::make_shared;
using std::shared_ptr;
using structure_utils::CircularBuffer;
using threading_utils::Semaphore;

namespace design_utils {

template <class BatchType>
class SharedBatchesProducer {
  protected:
    CircularBuffer<shared_ptr<BatchType>> batches;
    Semaphore start_semaphore, finish_semaphore;

  public:
    SharedBatchesProducer(const unsigned int max_batches):
        start_semaphore(max_batches - 1),
        finish_semaphore(0),
        batches(max_batches) {
      for (unsigned int i = 0; i < batches.size(); ++i) {
        batches.set(i, get_default_value());
      }
    }

  public:
    virtual ~SharedBatchesProducer(){};

    auto read_and_generate() -> void {
      for (unsigned int batch_id = 0; continue_read_condition(); ++batch_id) {
        do_at_batch_start(batch_id);
        generate();
        do_at_batch_finish(batch_id);
      }
      do_at_generate_finish();
    }

    auto operator>>(shared_ptr<BatchType> &out) -> bool {
      start_semaphore.release();
      finish_semaphore.acquire();
      if (batches.empty()) { return false; }
      out = batches.current_read();
      batches.step_read();
      return true;
    }

  protected:
    auto virtual get_default_value() -> shared_ptr<BatchType> {
      return make_shared<BatchType>();
    };
    auto virtual continue_read_condition() -> bool = 0;
    auto virtual do_at_batch_start(unsigned int batch_id = 0) -> void {
      start_semaphore.acquire();
    }
    auto virtual generate() -> void = 0;
    auto virtual do_at_batch_finish(unsigned int batch_id = 0) -> void {
      batches.step_write();
      finish_semaphore.release();
    }
    auto virtual do_at_generate_finish() -> void { finish_semaphore.release(); }
};

}  // namespace design_utils
#endif
