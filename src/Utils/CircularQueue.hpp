#include <vector>

using std::vector;

namespace utils {

template <class T>
class CircularQueue {
  private:
    vector<T> q;
    size_t front_idx = 0, back_idx = 0;

  public:
    CircularQueue(size_t capcity): q(capcity + 1) {}

    auto front() -> T & { return q[front_idx]; }

    auto pop() -> void { front_idx = (front_idx + 1) % capacity(); }

    auto push(T element) -> void {
      q[back_idx] = element;
      back_idx = (back_idx + 1) % q.size();
    }

    auto empty() -> bool { return front_idx == back_idx; }

    auto full() -> bool { return (back_idx + 1) % q.size() == front_idx; }

    auto size() -> size_t {
      if (front_idx > back_idx) { return q.size() - (front_idx - back_idx); }
      return back_idx - front_idx;
    }

    auto capacity() -> size_t { return q.size() - 1; }
};

}
