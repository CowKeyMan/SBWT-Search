#ifndef RANK_TEST_H
#define RANK_TEST_H

/**
 * @file Rank_test.h
 * @brief Header for functions used in testing the device rank function
 */

#include "Tools/GpuPointer.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using gpu_utils::GpuPointer;

auto get_rank(
  const GpuPointer<u64> &bit_vector,
  const GpuPointer<u64> &poppy_layer_0,
  const GpuPointer<u64> &poppy_layer_1_2,
  u64 index
) -> u64;

}  // namespace sbwt_search

#endif
