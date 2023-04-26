#include "Global/GlobalDefinitions.h"
#include "Presearcher/Presearcher.cuh"
#include "Presearcher/Presearcher.h"
#include "Tools/BitDefinitions.h"
#include "Tools/GpuUtils.h"
#include "hip/hip_runtime.h"

namespace sbwt_search {

auto Presearcher::launch_presearch_kernel(
  unique_ptr<GpuPointer<u64>> &presearch_left,
  unique_ptr<GpuPointer<u64>> &presearch_right,
  u64 blocks_per_grid
) -> void {
  hipLaunchKernelGGL(
    d_presearch,
    blocks_per_grid,
    threads_per_block,
    0,
    nullptr,
    container->get_c_map().data(),
    container->get_acgt_pointers().data(),
    container->get_layer_0_pointers().data(),
    container->get_layer_1_2_pointers().data(),
    presearch_left->data(),
    presearch_right->data()
  );
  GPU_CHECK(hipPeekAtLastError());
  GPU_CHECK(hipDeviceSynchronize());
}

}  // namespace sbwt_search
