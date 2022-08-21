#include <chrono>
#include <memory>
#include <thread>

#include "gtest/gtest.h"

#include "Utils/BoundedSemaphore.hpp"
#include "Utils/Semaphore_test.hpp"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::this_thread::sleep_for;

const auto sleep_amount = 100;

namespace threading_utils {

TEST(BoundedSemaphoreTest, TestBasic) {
  int counter = 0, counter_1, counter_2;
  BoundedSemaphore sem(2, 3);

#pragma omp parallel sections
  {
#pragma omp section
    {
      sem.release();
      ++counter;
      sem.release();
      ++counter;
      sem.release();
      ++counter;
    }
#pragma omp section
    {
      sleep_for(milliseconds(sleep_amount));
      counter_1 = counter;
      sem.acquire();
      sleep_for(milliseconds(sleep_amount));
      counter_2 = counter;
      sem.acquire();
    }
  }
  ASSERT_EQ(1, counter_1);
  ASSERT_EQ(2, counter_2);
  ASSERT_EQ(3, counter);
}

TEST(BoundedSemaphoreTest, SemaphoreBasic) {
  semaphore_basic_test<BoundedSemaphore>();
}

TEST(BoundedSemaphoreTest, Start0) {
  semaphore_test_start_0<BoundedSemaphore>();
}

TEST(BoundedSemaphoreTest, OverReleased) {
  BoundedSemaphore sem(4, 1);
  milliseconds::rep time_taken;

#pragma omp parallel sections
  {
#pragma omp section
    {
      for (int i = 0; i < 3; ++i) { sem.acquire(); }
      sleep_for(milliseconds(sleep_amount));
      sem.acquire();
    }
#pragma omp section
    {
      auto start_time = high_resolution_clock::now();
      sem.release();
      auto end_time = high_resolution_clock::now();
      time_taken = duration_cast<milliseconds>(end_time - start_time).count();
    }
  }
  ASSERT_GE(time_taken, sleep_amount);
}

}  // namespace threading_utils
