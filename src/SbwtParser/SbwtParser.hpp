#ifndef SBWT_PARSER_HPP
#define SBWT_PARSER_HPP

/**
 * @file SbwtParser.hpp
 * @brief Contains functions for interacting and reading SBWT index files
 * */

#include <cstddef>
#include <istream>
#include <memory>
#include <string>
#include <vector>

#include <sdsl/bit_vectors.hpp>

#include "Builder/Builder.h"
#include "SbwtContainer/CpuSbwtContainer.hpp"
#include "Utils/BitVectorUtils.h"
#include "Utils/IOUtils.h"
#include "Utils/TypeDefinitions.h"

using sdsl::bit_vector;
using std::istream;
using std::shared_ptr;
using std::string;
using std::vector;

namespace sbwt_search {

template <class Implementation, class Container>
class SbwtParser: private Builder {
  private:
    Implementation *const host;

  protected:
    SbwtParser(): host(static_cast<Implementation *>(this)) {}

  public:
    shared_ptr<Container> parse() {
      check_if_has_built();
      return host->do_parse();
    }
};

class SdslSbwtParser: public SbwtParser<SdslSbwtParser, SdslSbwtContainer> {
    friend SbwtParser;

  private:
    const string filename;

    shared_ptr<SdslSbwtContainer> do_parse();
    void assert_plain_matrix(istream &in) const;

  public:
    SdslSbwtParser(const string filename): filename(filename), SbwtParser() {}
};

class BitVectorSbwtParser:
    public SbwtParser<BitVectorSbwtParser, BitVectorSbwtContainer> {
    friend SbwtParser;

  private:
    const string files_prefix;
    u64 bits_total;

    shared_ptr<BitVectorSbwtContainer> do_parse();
    vector<vector<u64>> parse_acgt();
    vector<u64> parse_single_acgt(string filename);

  public:
    BitVectorSbwtParser(const string files_prefix):
        files_prefix(files_prefix), SbwtParser() {}
};

}

#endif
