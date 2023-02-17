#include <gtest/gtest.h>

#include "ColorIndexBuilder/ColorIndexBuilder.h"

namespace sbwt_search {

TEST(ColorIndexBuilderTest, full) {
  auto host = ColorIndexBuilder("test_objects/themisto_example.tcolors");
  auto cpu_container = host.get_cpu_color_index_container();
  auto gpu_container = cpu_container.to_gpu();

  vector<u64> temp;
  // start asserstions
  // Assertions regarding dense arrays
  gpu_container.dense_arrays.copy_to(temp);
  EXPECT_EQ(temp.size(), 0);
  gpu_container.dense_arrays_intervals.copy_to(temp);
  EXPECT_EQ(temp.size(), 2);  // 1 for the end + the extra u64
  EXPECT_EQ(gpu_container.dense_arrays_intervals_width, 1);
  // Assertions regarding sparse arrays
  gpu_container.sparse_arrays.copy_to(temp);
  EXPECT_EQ(temp.size(), 2);  // there are 4 items + the extra u64
  EXPECT_EQ(gpu_container.sparse_arrays_width, 2);
  gpu_container.sparse_arrays_intervals.copy_to(temp);
  EXPECT_EQ(temp.size(), 2);  // there are 5 items + the extra u64
  EXPECT_EQ(gpu_container.sparse_arrays_intervals_width, 3);
  // Assertions regarding is_dense_marks
  gpu_container.is_dense_marks.copy_to(temp);
  EXPECT_EQ(temp.size(), 1);  // There are 4 bits, rounded up to a single u64
  gpu_container.is_dense_marks_poppy_layer_0.copy_to(temp);
  EXPECT_EQ(temp.size(), 1);
  EXPECT_EQ(temp[0], 0);
  gpu_container.is_dense_marks_poppy_layer_1_2.copy_to(temp);
  EXPECT_EQ(temp.size(), 1);
  EXPECT_EQ(temp[0], 0);
  // Assertions regarding core_kmer_marks
  gpu_container.core_kmer_marks.copy_to(temp);
  EXPECT_EQ(temp.size(), 10);  // There are 635 bits, rounded up to 10 u64s
  gpu_container.core_kmer_marks_poppy_layer_0.copy_to(temp);
  EXPECT_EQ(temp.size(), 1);
  gpu_container.core_kmer_marks_poppy_layer_1_2.copy_to(temp);
  EXPECT_EQ(temp.size(), 1);  // there are 136 total_1s in this poppy structure
  // Assertions regarding color_set_idxs
  gpu_container.color_set_idxs.copy_to(temp);
  EXPECT_EQ(temp.size(), 6);  // 5 items + the extra u64
  EXPECT_EQ(gpu_container.color_set_idxs_width, 2);
  // Others
  EXPECT_EQ(gpu_container.num_color_sets, 4);
}

}  // namespace sbwt_search
