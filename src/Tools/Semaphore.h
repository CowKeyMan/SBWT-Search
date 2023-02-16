#ifndef SEMAPHORE_H
#define SEMAPHORE_H

/**
 * @file Semaphore.h
 * @brief A C++ implementation for counted semaphores using standard library
 * mutex and condition variables
 */

// Credit for class base:
// http://www.cs.umd.edu/~shankar/412-Notes/10x-countingSemUsingBinarySem.pdf

#include "Tools/OmpLock.h"
#include "Tools/TypeDefinitions.h"

namespace threading_utils {

class Semaphore {
private:
  OmpLock acquire_gate, count_protector;
  u64 count;

public:
  explicit Semaphore(unsigned int starting_count = 1);
  auto acquire() -> void;
  auto release() -> void;
};

}  // namespace threading_utils

#endif
