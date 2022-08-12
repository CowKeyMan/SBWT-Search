#ifndef I_O_UTILS_HPP
#define I_O_UTILS_HPP

/**
 * @file IOUtils.hpp
 * @brief Contains utilities to ease interacting with IO streams
 *        such as check if a file exists
 * */

#include <fstream>
#include <iostream>
#include <stdexcept>

using std::ifstream;
using std::ofstream;
using std::runtime_error;
using std::string;

namespace sbwt_search {

class ThrowingIfstream: public ifstream {
  public:
    ThrowingIfstream(const string filename, ios_base::openmode mode):
        ifstream(filename, mode) {
      if (this->fail()) {
        throw runtime_error(
          string("The input file ") + filename + " cannot be opened"
        );
      }
    }
    static void check_file_exists(const string filename);
};

class ThrowingOfstream: public ofstream {
  public:
    ThrowingOfstream(const string filepath, ios_base::openmode mode):
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
    static void check_path_valid(const string filepath);
};

}

#endif
