#ifndef SEMAPHORE__TEST_H
#define SEMAPHORE__TEST_H

/**
 * @file Semaphore_test.h
 * @brief Since these tests will be run by both the semaphore and bounded
 * semaphore, they have been extracted to their own class and templated to be
 * used by both
 * */

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "Utils/Semaphore.hpp"

using std::chrono::milliseconds;
using std::this_thread::sleep_for;

namespace threading_utils {

const auto sleep_amount = 100;

template <class Sem>
void semaphore_basic_test() {
  int counter = 0, counter_1, counter_2;
  Sem sem(3);

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

template <class Sem>
void semaphore_test_start_0() {
  int counter = 0, counter_1;
  Sem sem(0);

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
      counter_1 = counter;
      while (counter < 5) { sem.release(); }
    }
  }

  ASSERT_EQ(0, counter_1);
}

}

#endif