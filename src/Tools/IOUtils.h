#ifndef I_O_UTILS_H
#define I_O_UTILS_H

/**
 * @file IOUtils.h
 * @brief Contains utilities to ease interacting with IO streams
 *        such as check if a file exists
 */

#include <bit>
#include <fstream>
#include <string>
#include <vector>

namespace io_utils {

using std::bit_cast;
using std::ifstream;
using std::ios;
using std::ofstream;
using std::string;
using std::vector;

class ThrowingIfstream: public ifstream {
public:
  ThrowingIfstream(const string &filename, ios::openmode mode);
  static void check_file_exists(const string &filename);

  auto read_string_with_size() -> string;
  template <class Real>
  auto read_real() -> Real {
    Real r = 0;
    read(bit_cast<char *>(&r), sizeof(Real));
    return r;
  }
};

class ThrowingOfstream: public ofstream {
public:
  ThrowingOfstream(const string &filepath, ios::openmode mode);
  static void check_path_valid(const string &filepath);

  using ofstream::write;

  template <class Real>
  auto write(Real r) -> void {
    write(bit_cast<char *>(&r), sizeof(r));
  }

  template <class T>
  auto write(const vector<T> &v) -> void {
    write(bit_cast<char *>(v.data()), v.size() * sizeof(T));
  }

  auto write_string_with_size(const string &s) -> void;
};

}  // namespace io_utils

#endif
