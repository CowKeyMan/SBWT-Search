#include <stdexcept>
#include <string>

#include "Builder/Builder.h"

using std::string;

namespace sbwt_search {

auto Builder::check_if_has_built() -> void {
  if (has_built) { throw std::logic_error("Already Built"); }
  has_built = true;
}

}
