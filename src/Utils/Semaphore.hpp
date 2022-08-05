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

using uint = unsigned int;

namespace threading_utils {

class Semaphore {
  private:
    omp_lock_t count_protector_, acquire_gate_;
    omp_lock_t *count_protector = &count_protector_,
               *acquire_gate = &acquire_gate_;
    uint count;

  public:
    Semaphore(uint starting_count = 1): count(starting_count) {
      omp_init_lock(count_protector);
      omp_init_lock(acquire_gate);
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
      if (previous_count == 0) { omp_unset_lock(acquire_gate); }
      omp_unset_lock(count_protector);
    }

    ~Semaphore() {
      omp_destroy_lock(count_protector);
      omp_destroy_lock(acquire_gate);
    }
};

}

#endif
