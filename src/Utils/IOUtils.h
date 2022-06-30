#ifndef I_O_UTILS_H
#define I_O_UTILS_H

/**
 * @file IOUtils.h
 * @brief Contains utilities to ease interacting with IO streams
 *        such as check if a file exists
 * */

#include <fstream>
#include <iostream>
#include <stdexcept>

using std::ios;
using std::ios_base;
using std::is_base_of;
using std::ostream;
using std::runtime_error;
using std::string;

namespace sbwt_search {

template <class BaseStream>
class ThrowingStream: public BaseStream {
public:
  ThrowingStream(const char *filename, ios_base::openmode mode):
    BaseStream(filename, mode) {
    static_assert(
      is_base_of<ios, BaseStream>::value, "BaseStream must inherit C++ stream"
    );
    if (this->fail()) {
      throw runtime_error(string("The file ") + filename + " cannot be opened");
    }
  }
};

void check_file_exists(const char *filename);

}

#endif
