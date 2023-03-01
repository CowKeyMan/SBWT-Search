#include <gtest/gtest.h>

#include "ColorIndexBuilder/ColorIndexBuilder.h"
#include "Global/GlobalDefinitions.h"

namespace sbwt_search {

TEST(ColorIndexBuilderTest, full) {
  auto host
    = ColorIndexBuilder("test_objects/themisto_example/GCA_combined_d1.tcolors"
    );
  auto cpu_container = host.get_cpu_color_index_container();
  auto gpu_container = cpu_container.to_gpu();

  vector<u64> temp;
  // start asserstions
  // Assertions regarding dense arrays
  gpu_container->dense_arrays.copy_to(temp);
  EXPECT_EQ(temp.size(), 1);
  gpu_container->dense_arrays_intervals.copy_to(temp);
  EXPECT_EQ(temp.size(), 2);  // 1 for the end + the extra u64
  EXPECT_EQ(gpu_container->dense_arrays_intervals_width, 2);
  // Assertions regarding sparse arrays
  gpu_container->sparse_arrays.copy_to(temp);
  EXPECT_EQ(temp.size(), 2);  // there are 9 items (18 bits) + the extra u64
  EXPECT_EQ(gpu_container->sparse_arrays_width, 2);
  gpu_container->sparse_arrays_intervals.copy_to(temp);
  EXPECT_EQ(temp.size(), 2);  // there are 4 items (16 bits) + the extra u64
  EXPECT_EQ(gpu_container->sparse_arrays_intervals_width, 4);
  // Assertions regarding is_dense_marks
  gpu_container->is_dense_marks.copy_to(temp);
  EXPECT_EQ(temp.size(), 1);  // There are 7 bits, rounded up to a single u64
  gpu_container->is_dense_marks_poppy_layer_0.copy_to(temp);
  EXPECT_EQ(temp.size(), 1);
  gpu_container->is_dense_marks_poppy_layer_1_2.copy_to(temp);
  EXPECT_EQ(temp.size(), 1);
  // Assertions regarding core_kmer_marks
  gpu_container->core_kmer_marks.copy_to(temp);
  const u64 core_kmer_marks_bits = 19474194;
  EXPECT_EQ(temp.size(), (core_kmer_marks_bits + u64_bits - 1) / u64_bits);
  gpu_container->core_kmer_marks_poppy_layer_0.copy_to(temp);
  EXPECT_EQ(temp.size(), 1);
  gpu_container->core_kmer_marks_poppy_layer_1_2.copy_to(temp);
  EXPECT_EQ(
    temp.size(), (core_kmer_marks_bits + superblock_bits - 1) / superblock_bits
  );
  // Assertions regarding color_set_idxs
  gpu_container->color_set_idxs.copy_to(temp);
  EXPECT_EQ(temp.size(), (19443822UL * 3 + u64_bits - 1) / u64_bits + 1);
  EXPECT_EQ(gpu_container->color_set_idxs_width, 3);
  // Others
  EXPECT_EQ(gpu_container->num_color_sets, 7);
  EXPECT_EQ(gpu_container->num_colors, 3);
}

}  // namespace sbwt_search
