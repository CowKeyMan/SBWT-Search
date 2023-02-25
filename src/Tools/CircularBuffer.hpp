#ifndef CIRCULAR_BUFFER_HPP
#define CIRCULAR_BUFFER_HPP

/**
 * @file CircularBuffer.hpp
 * @brief Implementation of a Circular Buffer which is used for optimising
 * memory when using large buffers
 */

#include <cstddef>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace structure_utils {

using std::vector;

template <class T>
class CircularBuffer {
private:
  vector<T> q;
  u64 read_idx = 0, write_idx = 0, population = 0;

public:
  explicit CircularBuffer(u64 size): q(size) {}
  CircularBuffer(u64 size, T default_value): q(size) {
    for (auto &x : q) { x = default_value; }
  }
  auto set(u64 idx, T value) { q[idx] = value; }
  auto get(u64 idx) -> T { return q[idx]; }

  auto current_read() const -> const T & { return q[read_idx]; }
  auto current_write() -> T & { return q[write_idx]; }

  auto step_write() -> void {
    write_idx = (write_idx + 1) % q.size();
    --population;
  }
  auto step_read() -> void {
    read_idx = (read_idx + 1) % q.size();
    ++population;
  }

  [[nodiscard]] auto size() const -> u64 { return population; }
  [[nodiscard]] auto capacity() const -> u64 { return q.size(); }
  [[nodiscard]] auto empty() const -> bool { return population == 0; }
};

}  // namespace structure_utils

#endif
