#ifndef SBWT_PARSER_HPP
#define SBWT_PARSER_HPP

/**
 * @file SbwtParser.hpp
 * @brief Contains functions for interacting and reading SBWT index files
 * */

#include <istream>
#include <memory>
#include <string>

#include "SbwtContainer/CpuSbwtContainer.hpp"

using std::istream;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

class SbwtParser {
  private:
    const string filename;
    void assert_plain_matrix(istream &in) const;

  public:
    SbwtParser(const string filename): filename(filename) {}
    shared_ptr<SdslSbwtContainer> parse();
};

}  // namespace sbwt_search

#endif
