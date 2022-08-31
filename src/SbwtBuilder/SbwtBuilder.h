#ifndef SBWT_BUILDER
#define SBWT_BUILDER

#include <istream>
#include <memory>
#include <string>

namespace sbwt_search {
class CpuSbwtContainer;
}  // namespace sbwt_search

using std::istream;
using std::string;
using std::unique_ptr;

namespace sbwt_search {

class SbwtBuilder {
  private:
    string filename;

  public:
    SbwtBuilder(string &filename): filename(filename) {}
    auto get_cpu_sbwt(bool build_index = true) -> unique_ptr<CpuSbwtContainer>;

  private:
    auto assert_plain_matrix(istream &in) const -> void;
    auto build_poppy(CpuSbwtContainer *container) -> void;
};

}  // namespace sbwt_search
#endif
