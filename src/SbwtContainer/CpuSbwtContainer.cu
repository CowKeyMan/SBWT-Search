#include <memory>
#include <utility>

#include "SbwtContainer/CpuSbwtContainer.hpp"
#include "SbwtContainer/GpuSbwtContainer.cuh"

using std::make_shared;
using std::shared_ptr;

namespace sbwt_search {

template <class CpuContainer>
shared_ptr<GpuSbwtContainer> build_gpu_container(CpuContainer &container) {
  auto result = make_shared<GpuSbwtContainer>(
    container.get_acgt(static_cast<ACGT>(0)),
    container.get_acgt(static_cast<ACGT>(1)),
    container.get_acgt(static_cast<ACGT>(2)),
    container.get_acgt(static_cast<ACGT>(3)),
    container.get_bits_total(),
    container.get_bit_vector_size()
  );

  auto layer_0 = container.get_layer_0();
  if (layer_0.size() > 0 && layer_0[0].size() > 0) {
    result->set_c_map(container.get_c_map());
    result->set_layer_0(container.get_layer_0());
    result->set_layer_1_2(container.get_layer_1_2());
  }
  return result;
}

shared_ptr<GpuSbwtContainer> SdslSbwtContainer::to_gpu() {
  return build_gpu_container<decltype(*this)>(*this);
}

shared_ptr<GpuSbwtContainer> BitVectorSbwtContainer::to_gpu() {
  return build_gpu_container<decltype(*this)>(*this);
}

}
