#include <chrono>
#include <memory>
#include <omp.h>
#include <thread>

#include "gtest/gtest.h"

#include "Tools/Semaphore.h"

using std::chrono::milliseconds;
using std::this_thread::sleep_for;

const auto sleep_amount = 100;

namespace threading_utils {

const auto expected_counter = 5;

TEST(SemaphoreTest, Basic) {  // NOLINT
  int counter = 0;
  int counter_1 = -1;
  int counter_2 = -1;
  Semaphore sem(3);
#pragma omp parallel sections num_threads(2)
  {
#pragma omp section
    {
#pragma unroll
      while (counter < expected_counter) {
        sem.acquire();
        counter += 1;
      }
    }
#pragma omp section
    {
      sleep_for(milliseconds(sleep_amount));
      counter_1 = counter;
      sem.release();
      sleep_for(milliseconds(sleep_amount));
      counter_2 = counter;
      sem.release();
    }
  }
  ASSERT_EQ(3, counter_1);
  ASSERT_EQ(4, counter_2);
  ASSERT_EQ(expected_counter, counter);
}

TEST(SemaphoreTest, start0) {
  int counter = 0;
  int counter_1 = -1;
  Semaphore sem(0);
#pragma omp parallel sections num_threads(2)
  {
#pragma omp section
    {
#pragma unroll
      while (counter < expected_counter) {
        sem.acquire();
        counter += 1;
      }
    }
#pragma omp section
    {
      counter_1 = counter;
#pragma unroll
      while (counter < expected_counter) { sem.release(); }
    }
  }
  ASSERT_EQ(0, counter_1);
}

}  // namespace threading_utils
