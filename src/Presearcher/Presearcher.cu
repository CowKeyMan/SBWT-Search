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
    0,
    container->get_c_map().get(),
    container->get_acgt_pointers().get(),
    container->get_layer_0_pointers().get(),
    container->get_layer_1_2_pointers().get(),
    presearch_left->get(),
    presearch_right->get()
  );
  GPU_CHECK(hipPeekAtLastError());
  GPU_CHECK(hipDeviceSynchronize());
}

}  // namespace sbwt_search
