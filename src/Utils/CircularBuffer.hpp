#ifndef CIRCULAR_BUFFER_HPP
#define CIRCULAR_BUFFER_HPP

/**
 * @file CircularBuffer.hpp
 * @brief Implementation of a Circular Buffer which is used for optimising
 * memory when using large buffers
 * */

#include <vector>

using std::vector;

namespace utils {

template <class T>
class CircularBuffer {
  private:
    vector<T> q;
    size_t read_idx = 0, write_idx = 0;

  public:
    CircularBuffer(size_t size, T default_value): q(size) {
      for (auto &x: q) { x = default_value; }
    }

    auto current_read() -> const T & { return q[read_idx]; }
    auto current_write() -> T & { return q[write_idx]; }

    auto step_write() -> void { write_idx = (write_idx + 1) % q.size(); }
    auto step_read() -> void { read_idx = (read_idx + 1) % q.size(); }

    auto size() -> size_t { return q.size(); }
    auto empty() -> bool { return write_idx == read_idx; }
};

}

#endif
