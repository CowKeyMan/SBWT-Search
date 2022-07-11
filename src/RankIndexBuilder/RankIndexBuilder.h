#ifndef RANK_INDEX_BUILDER_H
#define RANK_INDEX_BUILDER_H

/**
 * @file RankIndexBuilder.h
 * @brief Module responsible for building the rank index of the SBWT file
 *        Assumptions made:
 *          * basicblock_bits is a multiple of 64
 *          * superblock_bits is a multiple of basicblock_bits
 *          * hyperblock_bits is a multiple of hyperblock_bits
 * */

#include <cmath>
#include <cstddef>
#include <vector>

#include "Utils/TypeDefinitionUtils.h"

using std::vector;

namespace sbwt_search {

class RankIndexBuilder {
  private:
    const size_t bits_total;
    const u64 *bits_vector;
    const u64 basicblock_bits, superblock_bits, hyperblock_bits;
    vector<u64> layer_0, layer_1_2;

  public:
    RankIndexBuilder(
      const size_t bits_total,
      const u64 *bits_vector,
      const u64 superblock_bits = pow(2, 10),
      const u64 hyperblock_bits = pow(2, 32)
    ):
        bits_total(bits_total),
        bits_vector(bits_vector),
        superblock_bits(superblock_bits),
        basicblock_bits(superblock_bits / 4),
        hyperblock_bits(hyperblock_bits) {}
    void build_index();
    const vector<u64> &get_layer_0();
    const vector<u64> &get_layer_1_2();
};

}

#endif
