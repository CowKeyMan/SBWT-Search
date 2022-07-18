#include <vector>

#include "Rank/Rank.cuh"
#include "Utils/CudaUtils.cuh"
#include "Utils/TypeDefinitions.h"

using std::vector;

namespace sbwt_search {

__global__ void d_test_rank(
  const u64 *const data,
  const u64 *const L0,
  const u64 *const L12,
  const u64 i,
  u64 *out
) {
  *out = d_rank<1024, 1ULL << 32>(data, L0, L12, i);
}

auto get_rank_output(
  const vector<u64> &bit_vector,
  const vector<u64> &layer_0,
  const vector<u64> &layer_1_2,
  const vector<u64> &test_indexes
) -> vector<u64> {
  auto d_bit_vector = CudaPointer<u64>(bit_vector);
  auto d_layer_0 = CudaPointer<u64>(layer_0);
  auto d_layer_1_2 = CudaPointer<u64>(layer_1_2);
  auto d_output = CudaPointer<u64>(1);
  auto output = vector<u64>(test_indexes.size());
  for (auto i = 0; i < test_indexes.size(); ++i) {
    d_test_rank<<<1, 1>>>(
      d_bit_vector.get(),
      d_layer_0.get(),
      d_layer_1_2.get(),
      test_indexes[i],
      d_output.get()
    );
    d_output.copy_to(output[i]);
  }
  return output;
}

}
