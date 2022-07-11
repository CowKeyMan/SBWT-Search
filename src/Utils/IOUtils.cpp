#include <iostream>

#include "Utils/IOUtils.hpp"

namespace sbwt_search {

using std::ifstream;
using std::ios_base;

auto ThrowingIfstream::check_file_exists(const string filename) -> void {
  ThrowingIfstream(filename, ios_base::in);
}

auto ThrowingOfstream::check_path_valid(const string filepath) -> void {
  ThrowingOfstream(filepath, ios_base::in);
}

}
