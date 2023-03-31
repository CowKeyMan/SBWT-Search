#ifndef CPU_SBWT_CONTAINER_H
#define CPU_SBWT_CONTAINER_H

/**
 * @file CpuSbwtContainer.h
 * @brief SbwtContainer for that on the cpu side
 */

#include <memory>

#include "Poppy/Poppy.h"
#include "SbwtContainer/GpuSbwtContainer.h"
#include "SbwtContainer/SbwtContainer.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::shared_ptr;
using std::vector;

class CpuSbwtContainer: public SbwtContainer {
private:
  vector<vector<u64>> acgt;
  vector<Poppy> poppys;
  vector<u64> c_map;
  vector<u64> key_kmer_marks;

public:
  CpuSbwtContainer(
    vector<vector<u64>> &&acgt_,
    vector<Poppy> &&poppys_,
    vector<u64> &&c_map_,
    u64 num_bits,
    u64 bit_vector_size,
    u64 kmer_size,
    vector<u64> &&key_kmer_marks
  );
  [[nodiscard]] auto to_gpu() const -> shared_ptr<GpuSbwtContainer>;
};

}  // namespace sbwt_search

#endif
