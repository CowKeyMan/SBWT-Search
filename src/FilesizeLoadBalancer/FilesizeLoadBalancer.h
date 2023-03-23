#ifndef FILESIZE_LOAD_BALANCER_H
#define FILESIZE_LOAD_BALANCER_H

/**
 * @file FilesizeLoadBalancer.h
 * @brief Takes in a list of files and a number N. N determines how many
 * partitions (streams) we want for the files. The vector of files is then split
 * into N buckets of approximately equal size. This is known as the multiway
 * partitioning problem and more information about it can be found here:
 * https://en.wikipedia.org/wiki/Multiway_number_partitioning. In our
 * implementation, we use the greedy algorithm due to its simplicity and speed,
 * and it comes close to a bound of 4/3 of the optimal solution. This method is
 * also called the longest processing time first algorithm and more information
 * can be found here:
 * https://en.wikipedia.org/wiki/Longest-processing-time-first_scheduling. Other
 * methods were attempted but they were too slow, especially as N increases.
 */

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::map;
using std::pair;
using std::string;
using std::vector;

class FilesizeLoadBalancer {
private:
  vector<string> in_files, out_files;
  map<u64, vector<pair<string, string>>> size_to_files;

public:
  FilesizeLoadBalancer(
    const vector<string> &in_files_, const vector<string> &out_files_
  );
  auto partition(u64 partitions)
    -> pair<vector<vector<string>>, vector<vector<string>>>;

private:
  auto populate_size_to_files() -> void;
  auto get_smallest_partition_index(vector<u64> &partition_sizes) -> u64;
};

}  // namespace sbwt_search

#endif
