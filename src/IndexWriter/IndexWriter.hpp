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

template <class Implementation, class Container>
class IndexWriter {
  private:
    Implementation *const host;

  protected:
    const Container &container;
    IndexWriter(Container &container):
        host(static_cast<Implementation *>(this)), container(container) {}

  public:
    void write(const string path) const { host->do_write(path); }
};

class SdslIndexWriter: public IndexWriter<SdslIndexWriter, SdslSbwtContainer> {
    friend IndexWriter;

  private:
    const string format = "plain-matrix";

    void do_write(string path) const;

  public:
    SdslIndexWriter(SdslSbwtContainer &container): IndexWriter(container) {}
};

class BitVectorIndexWriter:
    public IndexWriter<BitVectorIndexWriter, BitVectorSbwtContainer> {
    friend IndexWriter;

  private:
    void do_write(string path) const;

  public:
    BitVectorIndexWriter(BitVectorSbwtContainer &container):
        IndexWriter(container) {}
};

}

#endif
