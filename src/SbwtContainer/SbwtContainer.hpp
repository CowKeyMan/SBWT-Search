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

class SdslSbwtContainer: public SbwtContainer<SdslSbwtContainer> {
    friend SbwtContainer;

  private:
    const vector<bit_vector> acgt;
    const bit_vector c_map;
    const u64 *do_get_acgt(ACGT letter);

  public:
    SdslSbwtContainer(
      const bit_vector &&a,
      const bit_vector &&c,
      const bit_vector &&g,
      const bit_vector &&t
    ):
        acgt{ a, c, g, t }, SbwtContainer(a.size(), a.capacity() / 64) {}

    bit_vector get_acgt_sdsl(ACGT letter) const;
};

class BitVectorSbwtContainer: public SbwtContainer<BitVectorSbwtContainer> {
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
        acgt{ a, c, g, t }, SbwtContainer(bits_total, a.size()) {}

    void change_acgt_endianness();
};

}

#endif
