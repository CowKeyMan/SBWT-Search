/* #include <gtest/gtest.h> */

/* #include "Searcher/Searcher.cuh" */

/* const auto kmer_size = 30; */
/* const auto presearch_letters = 12; */
/* const size_t superblock_bits = 1024; */
/* constexpr const size_t hyperblock_bits = 1ULL << 32;  // pow(2, 32); */
/* const auto threads_per_block = 1024; */
/* const auto reversed_bits = true; */

/* namespace sbwt_search { */

/* vector<u64> resulting_positions() { */
/*   auto result = vector<u64>(71); */
/*   for (int i = 0; i < result.size(); ++i) { x = 71 + i; } */
/* } */

/* TEST(SearchTest, SearchTestSdsl) { */
/*   // search_test_query.fna is the second line of search_test_indexed.fna */
/*   auto query_file_parser */
/*     = QueryFileParser("test_objects/search_test_query.fna", kmer_size); */
/*   query_file_parser.parse_kseqpp_streams(); */
/*   auto sequences_parser = RawSequencesParser<CharToBitsVector>( */
/*     query_file_parser.get_seqs(), */
/*     query_file_parser.get_total_positions(), */
/*     query_file_parser.get_total_letters(), */
/*     kmer_size */
/*   ); */
/*   sequences_parser.parse_serial(); */
/*   vector<u64> &positions = sequences_parser.get_positions(); */
/*   vector<u64> &bit_seqs = sequences_parser.get_bit_seqs(); */

/*   auto factory = BitVectorSdslFactory(); */
/*   // search_test_index.sbwt is the sbwt index of search_test_indexed.fna */
/*   auto sbwt_parser */
/*     = factory.get_sbwt_parser("test_objects/search_test_index.sbwt"); */
/*   auto cpu_container = sbwt_parser.parse(); */
/*   cpu_container.change_acgt_endianness(); */
/*   auto index_builder = CpuRankIndexBuilder< */
/*     decltype(cpu_container), */
/*     superblock_bits, */
/*     hyperblock_bits>(&cpu_container); */
/*   index_builder.build_index(); */

/*   auto gpu_container = cpu_container.to_gpu(); */
/*   auto presearcher = Presearcher(gpu_container.get()); */
/*   presearcher.presearch< */
/*     threads_per_block, */
/*     superblock_bits, */
/*     hyperblock_bits, */
/*     presearch_letters, */
/*     reversed_bits>(); */

/*   auto searcher = Searcher(gpu_container.get()); */

/*   auto result = searcher.search< */
/*     threads_per_block, */
/*     superblock_bits, */
/*     hyperblock_bits, */
/*     presearch_letters, */
/*     kmer_size>(positions, bit_seqs); */

/*   ASSERT_EQ(result, ); */
/* } */

/* } */
