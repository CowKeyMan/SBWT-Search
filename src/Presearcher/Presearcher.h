#ifndef PRESEARCHER_H
#define PRESEARCHER_H

/**
 * @file Presearcher.h
 * @brief The presearcher will search for k-mers of a certain size and cache
 * them, so that future searches can continue from this checkpoint
 */

#include <memory>

#include "SbwtContainer/GpuSbwtContainer.h"
#include "Tools/GpuPointer.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::shared_ptr;

class Presearcher {
private:
  shared_ptr<GpuSbwtContainer> container;

public:
  explicit Presearcher(shared_ptr<GpuSbwtContainer> container_);
  auto launch_presearch_kernel(
    unique_ptr<GpuPointer<u64>> &presearch_left,
    unique_ptr<GpuPointer<u64>> &presearch_right,
    size_t blocks_per_grid
  ) -> void;
  auto presearch() -> void;
};

}  // namespace sbwt_search

#endif
