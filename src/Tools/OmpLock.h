#ifndef OMP_LOCK_H
#define OMP_LOCK_H

/**
 * @file OmpLock.h
 * @brief A wrapper over omp_lock_t to allow deletion when the resource is
 * released, like a smart pointer
 */

#include <omp.h>

namespace threading_utils {

class OmpLock {
  omp_lock_t lock;

public:
  OmpLock();
  OmpLock(OmpLock &) = delete;
  OmpLock(OmpLock &&) = delete;
  auto operator=(OmpLock &) = delete;
  auto operator=(OmpLock &&) = delete;
  auto set_lock() -> void;
  auto unset_lock() -> void;
  ~OmpLock();
};

}  // namespace threading_utils

#endif
