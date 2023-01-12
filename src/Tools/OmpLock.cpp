#include "Tools/OmpLock.h"

namespace threading_utils {

// NOLINTNEXTLINE (cppcoreguidelines-pro-type-member-init,hicpp-member-init)
OmpLock::OmpLock() { omp_init_lock(&lock); }

auto OmpLock::set_lock() -> void { omp_set_lock(&lock); }
auto OmpLock::unset_lock() -> void { omp_unset_lock(&lock); }

OmpLock::~OmpLock() { omp_destroy_lock(&lock); }

}  // namespace threading_utils
