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

namespace io_utils {

using std::bit_cast;
using std::ifstream;
using std::ofstream;
using std::string;

class ThrowingIfstream: public ifstream {
public:
  ThrowingIfstream(const string &filename, ios_base::openmode mode);
  static void check_file_exists(const string &filename);
};

class ThrowingOfstream: public ofstream {
public:
  ThrowingOfstream(const string &filepath, ios_base::openmode mode);
  static void check_path_valid(const string &filepath);
};

auto read_string_with_size(ThrowingIfstream &is) -> string;
auto write_string_with_size(ThrowingOfstream &os, const string &s) -> void;

}  // namespace io_utils

#endif
