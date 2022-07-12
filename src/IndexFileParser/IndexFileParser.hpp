#ifndef INDEX_FILE_PARSER_HPP
#define INDEX_FILE_PARSER_HPP

/**
 * @file IndexFileParser.hpp
 * @brief Contains functions for interacting and reading SBWT index files
 * */

#include <cstddef>
#include <istream>
#include <memory>
#include <string>
#include <vector>

#include <sdsl/bit_vectors.hpp>

#include "Builder/Builder.h"
#include "SbwtContainer/SbwtContainer.hpp"
#include "Utils/BitVectorUtils.h"
#include "Utils/IOUtils.hpp"
#include "Utils/TypeDefinitionUtils.h"

using sdsl::bit_vector;
using std::istream;
using std::string;
using std::unique_ptr;
using std::vector;

namespace sbwt_search {

template <class Implementation, class Container>
class IndexFileParser: Builder {
  private:
    Implementation *const host;

  protected:
    IndexFileParser(): host(static_cast<Implementation *>(this)) {}

  public:
    Container parse() const { return host->do_parse(); }
};

class SdslIndexFileParser:
    public IndexFileParser<SdslIndexFileParser, SdslSbwtContainer> {
    friend IndexFileParser;

  private:
    const string filename;

    SdslSbwtContainer do_parse();
    void assert_plain_matrix(istream &in) const;

  public:
    SdslIndexFileParser(const string filename):
        filename(filename), IndexFileParser() {}
};

class BitVectorIndexFileParser:
    public IndexFileParser<BitVectorIndexFileParser, BitVectorSbwtContainer> {
    friend IndexFileParser;

  private:
    const string files_prefix;
    u64 bits_total;

    BitVectorSbwtContainer do_parse();
    vector<vector<u64>> parse_acgt();
    vector<u64> parse_single_acgt(string filename);

  public:
    BitVectorIndexFileParser(const string files_prefix):
        files_prefix(files_prefix), IndexFileParser() {}
};

}

#endif
