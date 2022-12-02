#include <memory>
#include <utility>

#include "SbwtContainer/GpuSbwtContainer.cuh"
#include "SbwtContainer/SbwtContainer.h"

using std::make_shared;
using std::shared_ptr;

namespace sbwt_search {

auto CpuSbwtContainer::to_gpu() -> shared_ptr<GpuSbwtContainer> {
  auto result = make_shared<GpuSbwtContainer>(
    get_acgt(ACGT::A),
    get_acgt(ACGT::C),
    get_acgt(ACGT::G),
    get_acgt(ACGT::T),
    num_bits,
    bit_vector_size,
    kmer_size
  );

  if (layer_0.size() > 0 && layer_0[0].size() > 0) {
    result->set_c_map(c_map);
    result->set_layer_0(layer_0);
    result->set_layer_1_2(layer_1_2);
  }
  return result;
}

}  // namespace sbwt_search
