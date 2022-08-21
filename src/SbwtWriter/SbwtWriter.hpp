#ifndef SBWT_WRITER_HPP
#define SBWT_WRITER_HPP

/**
 * @file SbwtWriter.hpp
 * @brief Writes the given SbwtContainer to disk
 */

#include <string>

namespace sbwt_search {
class BitVectorSbwtContainer;
class SdslSbwtContainer;
}  // namespace sbwt_search

using std::string;

namespace sbwt_search {

template <class Implementation, class Container>
class SbwtWriter {
  private:
    Implementation *const host;

  protected:
    const Container &container;
    const string path;

    SbwtWriter(const Container &container, const string path):
        container(container),
        path(path),
        host(static_cast<Implementation *>(this)) {}

  public:
    void write() const { host->do_write(); }
};

class SdslSbwtWriter: public SbwtWriter<SdslSbwtWriter, SdslSbwtContainer> {
    friend SbwtWriter;

  private:
    const string format = "plain-matrix";

    void do_write() const;

  public:
    SdslSbwtWriter(const SdslSbwtContainer &container, const string path):
        SbwtWriter(container, path) {}
};

class BitVectorSbwtWriter:
    public SbwtWriter<BitVectorSbwtWriter, BitVectorSbwtContainer> {
    friend SbwtWriter;

  private:
    void do_write() const;

  public:
    BitVectorSbwtWriter(
      const BitVectorSbwtContainer &container, const string path
    ):
        SbwtWriter(container, path) {}
};

}  // namespace sbwt_search

#endif
