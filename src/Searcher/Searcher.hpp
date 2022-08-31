#ifndef SEARCHER_HPP
#define SEARCHER_HPP

/**
 * @file Searcher.hpp
 * @brief Search implementation on cpu
 * */

#include <memory>
#include <vector>

#include "SbwtContainer/SbwtContainer.hpp"
#include "TestUtils/RankTestUtils.hpp"
#include "Utils/BitDefinitions.h"
#include "sdsl/bit_vectors.hpp"
#include "Utils/GlobalDefinitions.h"

using std::shared_ptr;
using std::vector;

namespace sbwt_search {

template <class Container>
class SearcherCpu {
  private:
    shared_ptr<Container> container;

  public:
    SearcherCpu(shared_ptr<Container> container): container(container) {}

    template <u32 kmer_size>
    vector<u64>
    search(const vector<u64> &kmer_positions, const vector<u64> &bit_seqs) {
      auto rank = dummy_cpu_rank;
      auto result = vector<u64>(kmer_positions.size());
      for (int i = 0; i < kmer_positions.size(); ++i) {
        const u64 kmer_index = kmer_positions[i] * 2;
        const u64 first_part = (bit_seqs[kmer_index / 64] << (kmer_index % 64));
        // shifting by 64 is undefined behaviour, so we must make sure to 0 out
        // the number instead
        const u64 second_part
          = (bit_seqs[kmer_index / 64 + 1] >> (64 - (kmer_index % 64)))
          & (-((kmer_index % 64) != 0));
        const u64 kmer = first_part | second_part;
        auto c = (kmer >> 62) & two_1s;
        auto node_left = container->get_c_map()[c];
        auto node_right = container->get_c_map()[c + 1];
        for (int j = 1; j < kmer_size; ++j) {
          c = (kmer >> (62 - j * 2)) & two_1s;
          auto previous_node_left = node_left;
          auto previous_node_right = node_right;
          node_left
            = container->get_c_map()[c]
            + rank(container->get_acgt(static_cast<ACGT>(c)), node_left);
          node_right
            = container->get_c_map()[c]
            + rank(container->get_acgt(static_cast<ACGT>(c)), node_right + 1)
            - 1;
        }
        if (node_left > node_right) node_left = -1ULL;
        result[i] = node_left;
      }
      return result;
    }
};

}  // namespace sbwt_search

#endif
