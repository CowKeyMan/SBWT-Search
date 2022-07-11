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
    IndexWriter(): host(static_cast<Implementation *>(this)) {}

  public:
    void write(const Container &container, const string path) const {
      host->do_write(container, path);
    }
};

class SdslIndexWriter: public IndexWriter<SdslIndexWriter, SdslSbwtContainer> {
    friend IndexWriter;

  private:
    const string format = "plain-matrix";

    void do_write(const SdslSbwtContainer &container, string path) const;

  public:
    SdslIndexWriter(): IndexWriter() {}
};

class BitVectorIndexWriter:
    public IndexWriter<BitVectorIndexWriter, BitVectorSbwtContainer> {
    friend IndexWriter;

  private:
    void do_write(const BitVectorSbwtContainer &container, string path) const;

  public:
    BitVectorIndexWriter(): IndexWriter() {}
};

}

#endif
