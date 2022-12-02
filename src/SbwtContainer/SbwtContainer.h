#ifndef SBWT_CONTAINER_H
#define SBWT_CONTAINER_H

/**
 * @file SbwtContainer.h
 * @brief Contains data class for SBWT index
 * */

#include <cstddef>
#include <memory>
#include <vector>

#include "Utils/TypeDefinitions.h"

using std::shared_ptr;
using std::size_t;
using std::unique_ptr;
using std::vector;

namespace sbwt_search {

enum class ACGT { A = 0, C = 1, G = 2, T = 3 };

class SbwtContainer {
  protected:
    const size_t bit_vector_size;
    const size_t num_bits;
    uint kmer_size;

    SbwtContainer(const size_t num_bits, const size_t bit_vector_size):
        num_bits(num_bits), bit_vector_size(bit_vector_size) {}

  public:
    auto get_bit_vector_size() const -> u64;
    auto get_bits_total() const -> u64;

  public:
    virtual ~SbwtContainer(){};
};

class GpuSbwtContainer;

class CpuSbwtContainer: public SbwtContainer {
  protected:
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
      unique_ptr<vector<u64>> &t
    );
    auto set_index(
      vector<u64> &&new_c_map,
      vector<vector<u64>> &&new_layer_0,
      vector<vector<u64>> &&new_layer_1_2
    ) -> void;
    auto get_layer_0() const -> const vector<vector<u64>>;
    auto get_layer_0(ACGT letter) const -> const vector<u64>;
    auto get_layer_1_2() const -> const vector<vector<u64>>;
    auto get_layer_1_2(ACGT letter) const -> const vector<u64>;
    auto get_c_map() const -> const vector<u64>;
    auto to_gpu() -> shared_ptr<GpuSbwtContainer>;
    auto get_acgt(ACGT letter) const -> const u64 *;
    auto get_kmer_size() const -> uint;
};

}  // namespace sbwt_search

#endif
