#ifndef GLOBAL_DEFINITIONS_H
#define GLOBAL_DEFINITIONS_H

/**
 * @file GlobalDefinitions.h
 * @brief Contains all items which are constant throughout the program after
 * compilation
 */

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

constexpr const u64 hyperblock_bits = 1ULL << 32ULL;
constexpr const u64 superblock_bits = 1024;
constexpr const u64 basicblock_bits = 256;
constexpr const u64 presearch_letters = 12;
constexpr const uint threads_per_block = 1024;

}  // namespace sbwt_search

#endif
