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

#define popcount __builtin_popcountll

#include "SbwtContainer/SbwtContainer.hpp"
#include "Utils/TypeDefinitionUtils.h"

namespace sbwt_search {

template <
  class Implementation,
  class Container,
  u64 superblock_bits,
  u64 hyperblock_bits>
class RankIndexBuilder {
  private:
    Implementation *const host;

  protected:
    Container &container;
    const u64 basicblock_bits;
    RankIndexBuilder(Container &container):
        container(container),
        basicblock_bits(superblock_bits / 4),
        host(static_cast<Implementation *>(this)) {
      static_assert(
        (superblock_bits / 4) % 64 == 0 && hyperblock_bits % 64 == 0,
        "Block bits must be divisible by 64"
      );
      static_assert(
        superblock_bits < hyperblock_bits,
        "superblock_bits must be smaller than hyperblock_bits"
      );
    }

  public:
    void build_index() { host->do_build_index(); };
};

}

#endif
