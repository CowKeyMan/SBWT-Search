#include <bit>
#include <ios>

#include "Tools/IOUtils.h"
#include "Tools/TypeDefinitions.h"
#include "fmt/core.h"

namespace io_utils {

using fmt::format;
using std::bit_cast;
using std::ifstream;
using std::ios;
using std::string;

ThrowingIfstream::ThrowingIfstream(
  const string &filename, ios_base::openmode mode
):
  ifstream(filename, mode) {
  if (this->fail()) {
    throw ios::failure(format("The input file {} cannot be opened", filename));
  }
}

auto ThrowingIfstream::check_file_exists(const string &filename) -> void {
  ThrowingIfstream(filename, ios_base::in);
}

ThrowingOfstream::ThrowingOfstream(
  const string &filepath, ios_base::openmode mode
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

auto ThrowingOfstream::check_path_valid(const string &filepath) -> void {
  ThrowingOfstream(filepath, ios_base::out);
}

auto ThrowingOfstream::write_string_with_size(const string &s) -> void {
  u64 size = static_cast<u64>(s.size());
  std::ofstream::write(bit_cast<char *>(&size), sizeof(u64));
  (*this) << s;
}

auto ThrowingIfstream::read_string_with_size() -> string {
  u64 size = 0;
  read(bit_cast<char *>(&size), sizeof(u64));
  string s;
  s.resize(size);
  read(bit_cast<char *>(s.data()), static_cast<std::streamsize>(size));
  return s;
}

}  // namespace io_utils
