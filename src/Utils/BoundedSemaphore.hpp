#ifndef BOUNDED_SEMAPHORE_HPP
#define BOUNDED_SEMAPHORE_HPP

/**
 * @file BoundedSemaphore.hpp
 * @brief A sempahore with an upper bound for how much it can be released
 * */

#include <climits>

#include "Semaphore.hpp"

#include <iostream>
using namespace std;

namespace threading_utils {

using uint = unsigned int;

class BoundedSemaphore {
  private:
    Semaphore sem_main, sem_support;

  public:
    BoundedSemaphore(uint starting_count, uint maximum_count=UINT_MAX):
        sem_main(starting_count), sem_support(maximum_count - starting_count) {}

    void release() {
      sem_support.acquire();
      sem_main.release();
    }

    void acquire() {
      sem_main.acquire();
      sem_support.release();
    }
};

}

#endif
