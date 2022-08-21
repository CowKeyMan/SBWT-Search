#include <memory>

#include "Utils/Semaphore.hpp"
#include "Utils/Semaphore_test.hpp"
#include "gtest/gtest.h"

namespace threading_utils {

TEST(SemaphoreTest, Basic) { semaphore_basic_test<Semaphore>(); }

TEST(SemaphoreTest, tart0) { semaphore_test_start_0<Semaphore>(); }

}  // namespace threading_utils
