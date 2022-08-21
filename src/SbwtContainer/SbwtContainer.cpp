#include <byteswap.h>
#include <memory>

#include <ext/alloc_traits.h>
#include <sdsl/int_vector.hpp>

#include "SbwtContainer/CpuSbwtContainer.hpp"
#include "SbwtContainer/SbwtContainer.hpp"

using sdsl::bit_vector;

namespace sbwt_search {

SdslSbwtContainer::SdslSbwtContainer(
  const bit_vector &&a,
  const bit_vector &&c,
  const bit_vector &&g,
  const bit_vector &&t
):
    acgt{ a, c, g, t }, CpuSbwtContainer(a.size(), a.capacity() / 64) {}

auto SdslSbwtContainer::do_get_acgt(ACGT letter) const -> const u64 * {
  return acgt[static_cast<int>(letter)].data();
}

auto SdslSbwtContainer::get_acgt_sdsl(ACGT letter) const -> const bit_vector & {
  return acgt[static_cast<int>(letter)];
}

auto BitVectorSbwtContainer::do_get_acgt(ACGT letter) const -> const u64 * {
  return &acgt[static_cast<int>(letter)][0];
}

auto BitVectorSbwtContainer::change_acgt_endianness() -> void {
  for (auto &bit_vector: acgt) {
    for (auto &value: bit_vector) { value = bswap_64(value); }
  }
}

}  // namespace sbwt_search
