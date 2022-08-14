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
    size_t idx = 0;

  public:
    CircularBuffer(size_t capcity, T default_value): q(capcity) {
      for (auto &x: q) { x = default_value; }
    }

    auto current() -> T&  { return q[idx]; }

    auto step_forward() -> bool { idx = (idx + 1) % q.size(); }

    auto size() -> size_t { return q.size(); }
};

}

#endif
