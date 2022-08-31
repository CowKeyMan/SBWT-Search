#ifndef SBWT_CONTAINER_HPP
#define SBWT_CONTAINER_HPP

/**
 * @file SbwtContainer.hpp
 * @brief Contains data class for SBWT index
 * */

#include <cstddef>
#include <memory>
#include <vector>

#include <sdsl/int_vector.hpp>

#include "Utils/TypeDefinitions.h"

using sdsl::bit_vector;
using std::shared_ptr;
using std::size_t;
using std::unique_ptr;
using std::vector;

namespace sbwt_search {

enum class ACGT { A = 0, C = 1, G = 2, T = 3 };

class SbwtContainer {
  protected:
    const size_t bit_vector_size;
    const size_t bits_total;

    SbwtContainer(const size_t bits_total, const size_t bit_vector_size):
        bits_total(bits_total), bit_vector_size(bit_vector_size) {}

  public:
    auto get_bit_vector_size() const -> u64;
    auto get_bits_total() const -> u64;

  public:
    virtual ~SbwtContainer(){};
};

class GpuSbwtContainer;

class CpuSbwtContainer: public SbwtContainer {
  protected:
    vector<unique_ptr<bit_vector>> acgt;
    vector<vector<u64>> layer_0;
    vector<vector<u64>> layer_1_2;
    vector<u64> c_map;

  public:
    CpuSbwtContainer(
      unique_ptr<bit_vector> &a,
      unique_ptr<bit_vector> &c,
      unique_ptr<bit_vector> &g,
      unique_ptr<bit_vector> &t
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
};

}  // namespace sbwt_search

#endif
