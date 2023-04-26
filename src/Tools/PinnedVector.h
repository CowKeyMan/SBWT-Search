#ifndef PINNED_VECTOR_H
#define PINNED_VECTOR_H

/**
 * @file PinnedVector.h
 * @brief This class is part of the gpu utilities and represents an array of
 * fixed size but with some nice utilities to interact with other gpu utilities.
 * It uses pinned memory so that memory transfers between cpu and gpu are much
 * faster.
 */

#include "Tools/TypeDefinitions.h"

namespace gpu_utils {

template <class T>
class PinnedVector {
  T *ptr;
  u64 bytes;
  u64 num_elems = 0;

public:
  explicit PinnedVector(u64 size);
  auto data() const -> T *;
  auto operator[](u64 n) -> T &;
  auto operator[](const u64 n) const -> const T &;
  auto push_back(const T &elem) -> void;
  [[nodiscard]] auto size() const -> u64;
  auto resize(u64 n) -> void;
  auto empty() const -> bool;
};

}  // namespace gpu_utils

#endif
