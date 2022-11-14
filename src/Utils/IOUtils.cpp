#include <ios>

#include "Utils/IOUtils.h"
#include "fmt/core.h"

using fmt::format;
using std::ios;
using std::string;

namespace io_utils {

using std::ifstream;
using std::ios_base;

ThrowingIfstream::ThrowingIfstream(
  const string filename, ios_base::openmode mode
):
    ifstream(filename, mode) {
  if (this->fail()) {
    throw ios::failure(format("The input file {} cannot be opened", filename));
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
    throw ios::failure(fmt::format(
      "The path {}"
      " cannot be opened. Check that all the folders in the path is "
      "correct and that you have permission to create files in this path "
      "folder",
      filepath
    ));
  }
}

auto ThrowingOfstream::check_path_valid(const string filepath) -> void {
  ThrowingOfstream(filepath, ios_base::in);
}

}  // namespace io_utils
