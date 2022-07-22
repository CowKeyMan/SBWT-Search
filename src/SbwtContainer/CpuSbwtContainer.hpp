#ifndef CPU_SBWT_CONTAINER_HPP
#define CPU_SBWT_CONTAINER_HPP

/**
 * @file CpuSbwtContainer.hpp
 * @brief Contains data class for SBWT index which reside on CPU
 * */

#include <cstddef>
#include <memory>
#include <vector>

#include <sdsl/int_vector.hpp>

#include "SbwtContainer/SbwtContainer.hpp"

using sdsl::bit_vector;
using std::size_t;
using std::unique_ptr;

namespace sbwt_search {

class GpuSbwtContainer;

template <class Container>
class CpuSbwtContainer: public SbwtContainer<Container> {
  protected:
    vector<vector<u64>> layer_0;
    vector<vector<u64>> layer_1_2;
    vector<u64> c_map;

  protected:
    CpuSbwtContainer(const size_t bits_total, const size_t bit_vector_size):
        SbwtContainer<Container>(bits_total, bit_vector_size) {
      layer_0.resize(4);
      layer_1_2.resize(4);
      c_map.resize(5);
      c_map[0] = 1;
    }

  public:
    void set_layer_0(ACGT letter, vector<u64> &&new_layer_0) {
      layer_0[static_cast<int>(letter)] = new_layer_0;
    }
    void set_layer_1_2(ACGT letter, vector<u64> &&new_layer_1_2) {
      layer_1_2[static_cast<int>(letter)] = new_layer_1_2;
    }
    const vector<vector<u64>> &get_layer_0() const { return layer_0; }
    const vector<u64> &get_layer_0(ACGT letter) const {
      return layer_0[static_cast<int>(letter)];
    }
    const vector<vector<u64>> &get_layer_1_2() const { return layer_1_2; }
    const vector<u64> &get_layer_1_2(ACGT letter) const {
      return layer_1_2[static_cast<int>(letter)];
    }
    void set_c_map(vector<u64> &&value) { c_map = value; }
    const vector<u64> &get_c_map() const { return c_map; }
};

class SdslSbwtContainer: public CpuSbwtContainer<SdslSbwtContainer> {
    friend SbwtContainer;

  private:
    const vector<bit_vector> acgt;

    const u64 *do_get_acgt(ACGT letter) const;

  public:
    SdslSbwtContainer(
      const bit_vector &&a,
      const bit_vector &&c,
      const bit_vector &&g,
      const bit_vector &&t
    );

    bit_vector get_acgt_sdsl(ACGT letter) const;

    unique_ptr<GpuSbwtContainer> to_gpu();
};

class BitVectorSbwtContainer: public CpuSbwtContainer<BitVectorSbwtContainer> {
    friend SbwtContainer;

  private:
    vector<vector<u64>> acgt;

    const u64 *do_get_acgt(ACGT letter) const;

  public:
    BitVectorSbwtContainer(
      const vector<u64> &&a,
      const vector<u64> &&c,
      const vector<u64> &&g,
      const vector<u64> &&t,
      const size_t bits_total
    ):
        acgt{ a, c, g, t }, CpuSbwtContainer(bits_total, a.size()) {}

    void change_acgt_endianness();

    unique_ptr<GpuSbwtContainer> to_gpu();
};

}

#endif
