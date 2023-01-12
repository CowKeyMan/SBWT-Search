#ifndef CPU_SBWT_CONTAINER_H
#define CPU_SBWT_CONTAINER_H

/**
 * @file CpuSbwtContainer.h
 * @brief SbwtContainer for that on the cpu side
 */

#include <memory>

#include "SbwtContainer/GpuSbwtContainer.h"
#include "SbwtContainer/SbwtContainer.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::shared_ptr;
using std::vector;

class CpuSbwtContainer: public SbwtContainer {
private:
  vector<unique_ptr<vector<u64>>> acgt;
  vector<vector<u64>> layer_0;
  vector<vector<u64>> layer_1_2;
  vector<u64> c_map;

public:
  CpuSbwtContainer(
    u64 num_bits,
    unique_ptr<vector<u64>> &a,
    unique_ptr<vector<u64>> &c,
    unique_ptr<vector<u64>> &g,
    unique_ptr<vector<u64>> &t,
    uint _kmer_size
  );
  auto set_index(
    vector<u64> &&new_c_map,
    vector<vector<u64>> &&new_layer_0,
    vector<vector<u64>> &&new_layer_1_2
  ) -> void;
  auto get_layer_0() const -> const vector<vector<u64>> &;
  auto get_layer_0(ACGT letter) const -> const vector<u64> &;
  auto get_layer_1_2() const -> const vector<vector<u64>> &;
  auto get_layer_1_2(ACGT letter) const -> const vector<u64> &;
  auto get_c_map() const -> const vector<u64> &;
  auto to_gpu() -> shared_ptr<GpuSbwtContainer>;
  auto get_acgt(ACGT letter) const -> const u64 *;
};

}  // namespace sbwt_search

#endif
