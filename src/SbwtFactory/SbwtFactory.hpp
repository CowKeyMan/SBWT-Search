#ifndef SBWT_FACTORY_HPP
#define SBWT_FACTORY_HPP

/**
 * @file SbwtFactory.hpp
 * @brief Contains functions to create objects which are different
 *        based on the format being used
 */

#include "IndexFileParser/IndexFileParser.hpp"
#include "IndexWriter/IndexWriter.hpp"
#include "SbwtContainer/SbwtContainer.hpp"

namespace sbwt_search {

template <class Implementation, class Parser, class Container, class Writer>
class SbwtFactory {
  private:
    Implementation *const host;

  protected:
    SbwtFactory(): host(static_cast<Implementation *>(this)) {}

  public:
    Parser get_index_parser(string filepath) const {
      return host->do_get_index_parser(filepath);
    }
    Writer
    get_index_writer(const Container &container, const string path) const {
      return host->do_get_index_writer(container, path);
    }
};

class SdslSbwtFactory:
    public SbwtFactory<
      SdslSbwtFactory,
      SdslIndexFileParser,
      SdslSbwtContainer,
      SdslIndexWriter> {
    friend SbwtFactory;

  private:
    SdslIndexFileParser do_get_index_parser(string filepath) {
      return SdslIndexFileParser(filepath);
    }
    SdslIndexWriter
    do_get_index_writer(const SdslSbwtContainer &container, const string path) {
      return SdslIndexWriter(container, path);
    }
};

class BitVectorSbwtFactory:
    public SbwtFactory<
      BitVectorSbwtFactory,
      BitVectorIndexFileParser,
      BitVectorSbwtContainer,
      BitVectorIndexWriter> {
    friend SbwtFactory;

  private:
    BitVectorIndexFileParser do_get_index_parser(string filepath) const {
      return BitVectorIndexFileParser(filepath);
    }
    BitVectorIndexWriter do_get_index_writer(
      const BitVectorSbwtContainer &container, const string path
    ) const {
      return BitVectorIndexWriter(container, path);
    }
};

}

#endif
