#ifndef SBWT_CONTAINER_HPP
#define SBWT_CONTAINER_HPP

/**
 * @file SbwtContainer.hpp
 * @brief Contains data class for SBWT index
 * */

#include <cstddef>
#include <vector>

#include <sdsl/bit_vectors.hpp>

#include "Utils/TypeDefinitionUtils.h"

using sdsl::bit_vector;
using std::size_t;
using std::vector;

namespace sbwt_search {

enum class ACGT { A = 0, C = 1, G = 2, T = 3 };

class GpuSbwtContainer;

template <class Implementation>
class SbwtContainer {
  private:
    Implementation *const host;

  protected:
    const size_t bit_vector_size;
    const size_t bits_total;

    SbwtContainer(const size_t bits_total, const size_t bit_vector_size):
        bits_total(bits_total),
        bit_vector_size(bit_vector_size),
        host(static_cast<Implementation *>(this)) {}

  public:
    const u64 *get_acgt(ACGT letter) const { return host->do_get_acgt(letter); }

    u64 get_bit_vector_size() const { return bit_vector_size; }
    u64 get_bits_total() const { return bits_total; }
    GpuSbwtContainer to_gpu();
};

class GpuSbwtContainer: public SbwtContainer<GpuSbwtContainer> {
    friend SbwtContainer;
    // TODO: cudaptr<u64> A, C, G, T ...
  private:
    GpuSbwtContainer do_to_gpu() const { return *this; };

  public:
    GpuSbwtContainer(): SbwtContainer(3, 4) {}
};

template <class T>
GpuSbwtContainer SbwtContainer<T>::to_gpu() {
  return GpuSbwtContainer();
}

template <class Container>
class CpuSbwtContainer: public SbwtContainer<Container> {
  private:
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
    const vector<u64> &get_layer_0(ACGT letter) const {
      return layer_0[static_cast<int>(letter)];
    }
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

    const u64 *do_get_acgt(ACGT letter);

  public:
    SdslSbwtContainer(
      const bit_vector &&a,
      const bit_vector &&c,
      const bit_vector &&g,
      const bit_vector &&t
    ):
        acgt{ a, c, g, t }, CpuSbwtContainer(a.size(), a.capacity() / 64) {}

    bit_vector get_acgt_sdsl(ACGT letter) const;
};

class BitVectorSbwtContainer: public CpuSbwtContainer<BitVectorSbwtContainer> {
    friend SbwtContainer;

  private:
    vector<vector<u64>> acgt;
    vector<vector<u64>> layer_0;
    vector<vector<u64>> layer_1_2;

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
};

}

#endif
