#include <memory>
#include <stdexcept>

#include "SbwtContainer/CpuSbwtContainer.h"
#include "SbwtContainer/GpuSbwtContainer.h"

namespace sbwt_search {

using std::make_shared;

CpuSbwtContainer::CpuSbwtContainer(
  vector<vector<u64>> &&acgt_,
  vector<Poppy> &&poppys_,
  vector<u64> &&c_map_,
  u64 num_bits,
  u64 bit_vector_size,
  u64 kmer_size
):
    SbwtContainer(num_bits, bit_vector_size, kmer_size),
    acgt(std::move(acgt_)),
    poppys(std::move(poppys_)),
    c_map(std::move(c_map_)) {
  if (acgt.size() != 4 || poppys.size() != 4 || c_map.size() != 5) {
    throw std::runtime_error(
      "Wrong acgt, poppy or cmap size in CpuSbwtContainer"
    );
  }
}

auto CpuSbwtContainer::to_gpu() const -> shared_ptr<GpuSbwtContainer> {
  auto result = make_shared<GpuSbwtContainer>(
    acgt, poppys, c_map, get_num_bits(), get_bit_vector_size(), get_kmer_size()
  );
  return result;
}

}  // namespace sbwt_search
