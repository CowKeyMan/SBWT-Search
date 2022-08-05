#ifndef BOUNDED_SEMAPHORE_HPP
#define BOUNDED_SEMAPHORE_HPP

/**
 * @file BoundedSemaphore.hpp
 * @brief A sempahore with an upper bound for how much it can be released
 * */

#include <climits>
#include <omp.h>

namespace threading_utils {

using uint = unsigned int;

class BoundedSemaphore {
  private:
    omp_lock_t count_protector_, acquire_gate_, release_gate_;
    omp_lock_t *count_protector = &count_protector_,
               *acquire_gate = &acquire_gate_, *release_gate = &release_gate_;

  public:
    uint count, maximum;

  public:
    BoundedSemaphore(uint starting_count=1, uint maximum=UINT_MAX):
        count(starting_count), maximum(maximum) {
      omp_init_lock(count_protector);
      omp_init_lock(acquire_gate);
      omp_init_lock(release_gate);
      if (starting_count == 0) {
        omp_set_lock(acquire_gate);
      } else if (starting_count == maximum) {
        omp_set_lock(release_gate);
      }
    }

    void acquire() {
      omp_set_lock(acquire_gate);
      omp_set_lock(count_protector);
      uint previous_count = count;
      --count;
      if (count > 0) { omp_unset_lock(acquire_gate); }
      if (previous_count == maximum) { omp_unset_lock(release_gate); }
      omp_unset_lock(count_protector);
    }

    void release() {
      omp_set_lock(release_gate);
      omp_set_lock(count_protector);
      uint previous_count = count;
      ++count;
      if (previous_count == 0) { omp_unset_lock(acquire_gate); }
      if (count < maximum) { omp_unset_lock(release_gate); }
      omp_unset_lock(count_protector);
    }

    ~BoundedSemaphore() {
      omp_destroy_lock(count_protector);
      omp_destroy_lock(acquire_gate);
      omp_destroy_lock(release_gate);
    }
};

}

#endif
