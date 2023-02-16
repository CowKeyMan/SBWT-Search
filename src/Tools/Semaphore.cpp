#include "Tools/Semaphore.h"

namespace threading_utils {

Semaphore::Semaphore(unsigned int starting_count): count(starting_count) {
  if (starting_count == 0) { acquire_gate.set_lock(); }
}

auto Semaphore::acquire() -> void {
  acquire_gate.set_lock();
  count_protector.set_lock();
  --count;
  if (count > 0) { acquire_gate.unset_lock(); }
  count_protector.unset_lock();
}

auto Semaphore::release() -> void {
  count_protector.set_lock();
  const u64 previous_count = count;
  ++count;
  count_protector.unset_lock();
  if (previous_count == 0) { acquire_gate.unset_lock(); }
}

}  // namespace threading_utils
