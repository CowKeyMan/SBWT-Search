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

class Semaphore {
  private:
    omp_lock_t acquire_gate_, count_protector_;
    omp_lock_t *acquire_gate = &acquire_gate_,
               *count_protector = &count_protector_;
    unsigned int count;

  public:
    Semaphore(unsigned int starting_count = 1): count(starting_count) {
      omp_init_lock(acquire_gate);
      omp_init_lock(count_protector);
      if (starting_count == 0) { omp_set_lock(acquire_gate); }
    }

    void acquire() {
      omp_set_lock(acquire_gate);
      omp_set_lock(count_protector);
      --count;
      if (count > 0) { omp_unset_lock(acquire_gate); }
      omp_unset_lock(count_protector);
    }

    void release() {
      omp_set_lock(count_protector);
      int previous_count = count;
      ++count;
      omp_unset_lock(count_protector);
      if (previous_count == 0) { omp_unset_lock(acquire_gate); }
    }

    ~Semaphore() {
      omp_destroy_lock(count_protector);
      omp_destroy_lock(acquire_gate);
    }
};

}  // namespace threading_utils

#endif
