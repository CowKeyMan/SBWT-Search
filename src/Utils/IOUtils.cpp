#include <stdexcept>

#include "Utils/IOUtils.h"

using std::runtime_error;
using std::string;

namespace sbwt_search {

using std::ifstream;
using std::ios_base;

ThrowingIfstream::ThrowingIfstream(
  const string filename, ios_base::openmode mode
):
    ifstream(filename, mode) {
  if (this->fail()) {
    throw runtime_error(
      string("The input file ") + filename + " cannot be opened"
    );
  }
}

auto ThrowingIfstream::check_file_exists(const string filename) -> void {
  ThrowingIfstream(filename, ios_base::in);
}

ThrowingOfstream::ThrowingOfstream(
  const string filepath, ios_base::openmode mode
):
    ofstream(filepath, mode) {
  if (this->fail()) {
    throw runtime_error(
      string("The path ") + filepath
      + " cannot be opened. Check that all the folders in the path is "
        "correct and that you have permission to create files in this path "
        "folder"
    );
  }
}

auto ThrowingOfstream::check_path_valid(const string filepath) -> void {
  ThrowingOfstream(filepath, ios_base::in);
}

}  // namespace sbwt_search
