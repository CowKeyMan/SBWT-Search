#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ext/alloc_traits.h>
#include <sdsl/int_vector.hpp>

#include "PoppyBuilder/PoppyBuilder.h"
#include "SbwtBuilder/SbwtBuilder.h"
#include "SbwtContainer/SbwtContainer.h"
#include "Utils/IOUtils.h"
#include "Utils/Logger.h"
#include "Utils/TypeDefinitions.h"

using io_utils::ThrowingIfstream;
using log_utils::Logger;
using std::make_unique;
using std::move;
using std::runtime_error;
using std::unique_ptr;

namespace sbwt_search {

auto SbwtBuilder::get_cpu_sbwt(bool build_index)
  -> unique_ptr<CpuSbwtContainer> {
  ThrowingIfstream stream(filename, std::ios::in);
  assert_plain_matrix(stream);
  vector<unique_ptr<bit_vector>> acgt(4);
  Logger::log_timed_event("SBWTRead", Logger::EVENT_STATE::START);
  for (int i = 0; i < 4; ++i) {
    acgt[i] = make_unique<bit_vector>();
    acgt[i]->load(stream);
  }
  Logger::log_timed_event("SBWTRead", Logger::EVENT_STATE::STOP);
  auto container
    = make_unique<CpuSbwtContainer>(acgt[0], acgt[1], acgt[2], acgt[3]);
  Logger::log_timed_event("Poppy", Logger::EVENT_STATE::START);
  if (build_index) { build_poppy(container.get()); }
  Logger::log_timed_event("Poppy", Logger::EVENT_STATE::STOP);
  return container;
}

auto SbwtBuilder::build_poppy(CpuSbwtContainer *container) -> void {
  vector<vector<u64>> layer_0(4), layer_1_2(4);
  vector<u64> c_map(5);
  c_map[0] = 1;
#pragma omp parallel for
  for (int i = 0; i < 4; ++i) {
    auto builder = PoppyBuilder(
      container->get_bits_total(), container->get_acgt(static_cast<ACGT>(i))
    );
    builder.build();
    layer_0[i] = builder.get_layer_0();
    layer_1_2[i] = builder.get_layer_1_2();
    c_map[i + 1] = builder.get_total_count();
  }
  for (int i = 0; i < 4; ++i) { c_map[i + 1] += c_map[i]; }
  container->set_index(move(c_map), move(layer_0), move(layer_1_2));
}

void SbwtBuilder::assert_plain_matrix(istream &stream) const {
  size_t size;
  stream.read(reinterpret_cast<char *>(&size), sizeof(u64));
  string variant(size, '\0');
  stream.read(reinterpret_cast<char *>(&variant[0]), size);
  if (variant != "plain-matrix") {
    throw runtime_error("Error input is not a plain-matrix SBWT");
  }
}

}  // namespace sbwt_search
