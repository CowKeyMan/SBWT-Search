#include <iostream>

#include "IOUtils.h"

namespace sbwt_search {

using std::ifstream;
using std::ios_base;

void check_file_exists(const char *filename) {
  ThrowingStream<ifstream> tmp(filename, ios_base::in);
}

}
