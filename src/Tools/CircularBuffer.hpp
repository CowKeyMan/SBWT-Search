#ifndef CIRCULAR_BUFFER_HPP
#define CIRCULAR_BUFFER_HPP

/**
 * @file CircularBuffer.hpp
 * @brief Implementation of a Circular Buffer which is used for optimising
 * memory when using large buffers
 */

#include <stddef.h>
#include <vector>

using std::vector;

namespace structure_utils {

template <class T>
class CircularBuffer {
private:
  vector<T> q;
  size_t read_idx = 0, write_idx = 0, population = 0;

public:
  CircularBuffer(size_t size): q(size) {}
  CircularBuffer(size_t size, T default_value): q(size) {
    for (auto &x : q) { x = default_value; }
  }
  auto set(size_t idx, T value) { q[idx] = value; }

  auto current_read() -> const T & { return q[read_idx]; }
  auto current_write() -> T & { return q[write_idx]; }

  auto step_write() -> void {
    write_idx = (write_idx + 1) % q.size();
    --population;
  }
  auto step_read() -> void {
    read_idx = (read_idx + 1) % q.size();
    ++population;
  }

  auto size() -> size_t { return population; }
  auto capacity() -> size_t { return q.size(); }
  auto empty() -> bool { return population == 0; }
};

}  // namespace structure_utils

#endif
