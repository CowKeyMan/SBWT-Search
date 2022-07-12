#ifndef SBWT_FACTORY_HPP
#define SBWT_FACTORY_HPP

/**
 * @file SbwtFactory.hpp
 * @brief Contains functions to create objects which are different
 *        based on the format being used
 */

#include "SbwtContainer/SbwtContainer.hpp"
#include "SbwtParser/SbwtParser.hpp"
#include "SbwtWriter/SbwtWriter.hpp"

namespace sbwt_search {

template <class Implementation, class Parser, class Container, class Writer>
class SbwtFactory {
  private:
    Implementation *const host;

  protected:
    SbwtFactory(): host(static_cast<Implementation *>(this)) {}

  public:
    Parser get_sbwt_parser(string filepath) const {
      return host->do_get_sbwt_parser(filepath);
    }
    Writer
    get_sbwt_writer(const Container &container, const string path) const {
      return host->do_get_sbwt_writer(container, path);
    }
};

class SdslSbwtFactory:
    public SbwtFactory<
      SdslSbwtFactory,
      SdslSbwtParser,
      SdslSbwtContainer,
      SdslSbwtWriter> {
    friend SbwtFactory;

  private:
    SdslSbwtParser do_get_sbwt_parser(string filepath) {
      return SdslSbwtParser(filepath);
    }
    SdslSbwtWriter
    do_get_sbwt_writer(const SdslSbwtContainer &container, const string path) {
      return SdslSbwtWriter(container, path);
    }
};

class BitVectorSbwtFactory:
    public SbwtFactory<
      BitVectorSbwtFactory,
      BitVectorSbwtParser,
      BitVectorSbwtContainer,
      BitVectorSbwtWriter> {
    friend SbwtFactory;

  private:
    BitVectorSbwtParser do_get_sbwt_parser(string filepath) const {
      return BitVectorSbwtParser(filepath);
    }
    BitVectorSbwtWriter do_get_sbwt_writer(
      const BitVectorSbwtContainer &container, const string path
    ) const {
      return BitVectorSbwtWriter(container, path);
    }
};

}

#endif
