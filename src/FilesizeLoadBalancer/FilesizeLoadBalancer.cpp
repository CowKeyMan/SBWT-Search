#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include "FilesizeLoadBalancer/FilesizeLoadBalancer.h"

namespace sbwt_search {

using std::runtime_error;
using std::filesystem::file_size;

FilesizeLoadBalancer::FilesizeLoadBalancer(
  const vector<string> &in_files_, const vector<string> &out_files_
):
    in_files(in_files_), out_files(out_files_) {
  if (in_files.size() != out_files.size()) {
    throw runtime_error("Input and output file sizes differ");
  }
  populate_size_to_files();
}

auto FilesizeLoadBalancer::partition(u64 partitions)
  -> pair<vector<vector<string>>, vector<vector<string>>> {
  vector<vector<string>> in_result(partitions);
  vector<vector<string>> out_result(partitions);
  vector<u64> partition_sizes(partitions, 0);
  // iterate map in reverse order
  // NOLINTNEXTLINE (modernize-loop-convert)
  for (auto iter = size_to_files.rbegin(); iter != size_to_files.rend();
       ++iter) {
    for (pair<string, string> &in_out : iter->second) {
      auto smallest_index = get_smallest_partition_index(partition_sizes);
      in_result[smallest_index].push_back(in_out.first);
      out_result[smallest_index].push_back(in_out.second);
      partition_sizes[smallest_index] += iter->first;
    }
  }
  return {in_result, out_result};
}

auto FilesizeLoadBalancer::populate_size_to_files() -> void {
  for (u64 i = 0; i < in_files.size(); ++i) {
    size_to_files[file_size(in_files[i])].emplace_back(
      std::make_pair(in_files[i], out_files[i])
    );
  }
}

auto FilesizeLoadBalancer::get_smallest_partition_index(
  vector<u64> &partition_sizes
) -> u64 {
  return std::min_element(partition_sizes.begin(), partition_sizes.end())
    - partition_sizes.begin();
}

}  // namespace sbwt_search
