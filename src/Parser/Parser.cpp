#include <stdexcept>
#include <string>

#include "Parser.h"

using std::string;

namespace sbwt_search {

auto Parser::check_if_has_parsed() -> void {
  if (has_parsed) { throw std::logic_error("Already Parsed"); }
  has_parsed = true;
}

}
