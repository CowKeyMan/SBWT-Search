#ifndef I_O_UTILS_H
#define I_O_UTILS_H

/**
 * @file IOUtils.h
 * @brief Contains utilities to ease interacting with IO streams
 *        such as check if a file exists
 * */

#include <fstream>
#include <string>

using std::ifstream;
using std::ofstream;
using std::string;

namespace sbwt_search {

class ThrowingIfstream: public ifstream {
  public:
    ThrowingIfstream(const string filename, ios_base::openmode mode);
    static void check_file_exists(const string filename);
};

class ThrowingOfstream: public ofstream {
  public:
    ThrowingOfstream(const string filepath, ios_base::openmode mode);
    static void check_path_valid(const string filepath);
};

}  // namespace sbwt_search

#endif
