#include "IOUtils.hpp"

#include <iostream>

namespace sbwt_search {

using std::ios_base;
using std::ifstream;

void check_file_exists (const char* filename) {
  ThrowingStream<ifstream> tmp(filename, ios_base::in);
}

}
