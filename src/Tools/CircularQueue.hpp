#ifndef CIRCULAR_QUEUE_HPP
#define CIRCULAR_QUEUE_HPP

/**
 * @file CircularQueue.hpp
 * @brief Implementation of a Circular Queue which is used for optimising
 * memory when using queues
 */

namespace structure_utils {

using std::vector;

template <class T>
class CircularQueue {
private:
  vector<T> q;
  u64 front_idx = 0, back_idx = 0;

public:
  CircularQueue(u64 capcity): q(capcity + 1) {}

  auto front() -> T & { return q[front_idx]; }

  auto pop() -> void { front_idx = (front_idx + 1) % q.size(); }

  auto push(T element) -> void {
    q[back_idx] = element;
    back_idx = (back_idx + 1) % q.size();
  }

  auto empty() -> bool { return front_idx == back_idx; }

  auto full() -> bool { return (back_idx + 1) % q.size() == front_idx; }

  auto size() -> u64 {
    if (front_idx > back_idx) { return q.size() - (front_idx - back_idx); }
    return back_idx - front_idx;
  }

  auto capacity() -> u64 { return q.size() - 1; }
};

}  // namespace structure_utils

#endif
