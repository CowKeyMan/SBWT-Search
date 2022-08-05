#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "Utils/Semaphore.hpp"

using std::chrono::milliseconds;
using std::this_thread::sleep_for;

const auto sleep_amount = 100;

namespace threading_utils {

TEST(SemaphoreTest, Test) {
  int counter = 0;
  Semaphore sem(3);
  int counter_1, counter_2;

#pragma omp parallel sections
  {
#pragma omp section
    {
      while (counter < 5) {
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
  ASSERT_EQ(5, counter);
}

}
