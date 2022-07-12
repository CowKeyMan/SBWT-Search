#ifndef INDEX_WRITER_HPP
#define INDEX_WRITER_HPP

/**
 * @file IndexWriter.hpp
 * @brief Writes the given SbwtContainer to disk
 */

#include <string>

#include "SbwtContainer/SbwtContainer.hpp"

using std::string;

namespace sbwt_search {

#include <iostream>
using namespace std;

template <class Implementation, class Container>
class IndexWriter {
  private:
    Implementation *const host;

  protected:
    const Container &container;
    const string path;

    IndexWriter(const Container &container, const string path):
        container(container),
        path(path),
        host(static_cast<Implementation *>(this)) {}

  public:
    void write() const { host->do_write(); }
};

class SdslIndexWriter: public IndexWriter<SdslIndexWriter, SdslSbwtContainer> {
    friend IndexWriter;

  private:
    const string format = "plain-matrix";

    void do_write() const;

  public:
    SdslIndexWriter(const SdslSbwtContainer &container, const string path):
        IndexWriter(container, path) {}
};

class BitVectorIndexWriter:
    public IndexWriter<BitVectorIndexWriter, BitVectorSbwtContainer> {
    friend IndexWriter;

  private:
    void do_write() const;

  public:
    BitVectorIndexWriter(
      const BitVectorSbwtContainer &container, const string path
    ):
        IndexWriter(container, path) {}
};

}

#endif
