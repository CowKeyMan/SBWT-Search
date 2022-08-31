#include <vector>

#include "Rank/Rank.cuh"
#include "Utils/CudaUtils.cuh"
#include "Utils/TypeDefinitions.h"
#include "Utils/GlobalDefinitions.h"

using gpu_utils::GpuPointer;
using std::vector;

namespace sbwt_search {

__global__ void d_test_rank(
  const u64 *const data,
  const u64 *const L0,
  const u64 *const L12,
  const u64 i,
  u64 *out
) {
  *out = d_rank(data, L0, L12, i);
}

auto get_rank_output(
  const vector<u64> &bit_vector,
  const vector<u64> &layer_0,
  const vector<u64> &layer_1_2,
  const vector<u64> &test_indexes
) -> vector<u64> {
  auto d_bit_vector = GpuPointer<u64>(bit_vector);
  auto d_layer_0 = GpuPointer<u64>(layer_0);
  auto d_layer_1_2 = GpuPointer<u64>(layer_1_2);
  auto d_output = GpuPointer<u64>(1);
  auto output = vector<u64>(test_indexes.size());
  for (auto i = 0; i < test_indexes.size(); ++i) {
    d_test_rank<<<1, 1>>>(
      d_bit_vector.get(),
      d_layer_0.get(),
      d_layer_1_2.get(),
      test_indexes[i],
      d_output.get()
    );
    d_output.copy_to(&output[i]);
  }
  return output;
}

auto get_rank_output(
  const vector<u64> &bit_vector,
  const vector<u64> &layer_0,
  const vector<u64> &layer_1_2,
  const vector<u64> &test_indexes
) -> vector<u64> {
  return get_rank_output<true>(bit_vector, layer_0, layer_1_2, test_indexes);
}

}  // namespace sbwt_search
