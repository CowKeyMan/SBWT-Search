#include <stdexcept>

#include "ColorIndexBuilder/ColorIndexBuilder.h"
#include "PoppyBuilder/PoppyBuilder.h"
#include "sdsl/int_vector.hpp"
#include "sdsl/rank_support.hpp"

namespace sbwt_search {

using std::ios;
using std::runtime_error;

ColorIndexBuilder::ColorIndexBuilder(const string &filename):
    in_stream(filename, ios::in | ios::binary) {
  string filetype = in_stream.read_string_with_size();
  if (filetype != "sdsl-hybrid-v4") {
    throw runtime_error(
      "The colors file has an incorrect format. Expected 'sdsl-hybrid-v4'"
    );
  }
}

auto ColorIndexBuilder::get_cpu_color_index_container()
  -> CpuColorIndexContainer {
  sdsl::bit_vector dense_arrays;
  sdsl::int_vector<> dense_arrays_intervals;
  sdsl::int_vector<> sparse_arrays;
  sdsl::int_vector<> sparse_arrays_intervals;
  sdsl::bit_vector is_dense_marks;
  Poppy is_dense_marks_poppy;
  sdsl::bit_vector core_kmer_marks;
  Poppy core_kmer_marks_poppy;
  sdsl::int_vector<> color_set_idxs;
  u64 num_color_sets = static_cast<u64>(-1);

  sdsl::rank_support_v5 discard;

  dense_arrays.load(in_stream);
  dense_arrays_intervals.load(in_stream);

  sparse_arrays.load(in_stream);
  sparse_arrays_intervals.load(in_stream);

  is_dense_marks.load(in_stream);
  discard.load(
    in_stream, &is_dense_marks
  );  // skip is_dense_marks rank structure
  is_dense_marks_poppy
    = PoppyBuilder(
        {is_dense_marks.data(), is_dense_marks.capacity() / u64_bits},
        is_dense_marks.size()
    )
        .get_poppy();

  core_kmer_marks.load(in_stream);
  discard.load(
    in_stream, &core_kmer_marks
  );  // skip is_dense_marks rank structure
  core_kmer_marks_poppy
    = PoppyBuilder(
        {core_kmer_marks.data(), core_kmer_marks.capacity() / u64_bits},
        core_kmer_marks.size()
    )
        .get_poppy();

  color_set_idxs.load(in_stream);

  num_color_sets = is_dense_marks.size();

  return {
    dense_arrays,
    dense_arrays_intervals,
    sparse_arrays,
    sparse_arrays_intervals,
    is_dense_marks,
    is_dense_marks_poppy,
    core_kmer_marks,
    core_kmer_marks_poppy,
    color_set_idxs,
    num_color_sets};
}

}  // namespace sbwt_search
