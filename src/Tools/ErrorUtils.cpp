#include <stdexcept>
#include <string>

#include "Tools/ErrorUtils.h"

using std::runtime_error;
using std::string;
using std::to_string;

namespace error_utils {

auto _throw_uninitialised(string file, unsigned int line) -> void {
  throw runtime_error(
    "Calling unintialized function at " + file + ":" + to_string(line)
  );
}

}  // namespace error_utils
