#ifndef SEMAPHORE_HPP
#define SEMAPHORE_HPP

/**
 * @file Semaphore.hpp
 * @brief A C++ implementation for counted semaphores using standard library
 * mutex and condition variables
 * */

// Credit for class base:
// http://www.cs.umd.edu/~shankar/412-Notes/10x-countingSemUsingBinarySem.pdf

#include <omp.h>

namespace threading_utils {

using uint = unsigned int;

class Semaphore {
  private:
    omp_lock_t acquire_gate_;
    omp_lock_t *acquire_gate = &acquire_gate_;
    uint count;

  public:
    Semaphore(uint starting_count = 1): count(starting_count) {
      omp_init_lock(acquire_gate);
      if (starting_count == 0) { omp_set_lock(acquire_gate); }
    }

    void acquire() {
      omp_set_lock(acquire_gate);
#pragma omp critical(SEMAPHORE_COUNT_PROTECTOR)
      {
        --count;
        if (count > 0) { omp_unset_lock(acquire_gate); }
      }
    }

    void release() {
#pragma omp critical(SEMAPHORE_COUNT_PROTECTOR)
      {
        int previous_count = count;
        ++count;
        if (previous_count == 0) { omp_unset_lock(acquire_gate); }
      }
    }

    ~Semaphore() { omp_destroy_lock(acquire_gate); }
};

}  // namespace threading_utils

#endif
