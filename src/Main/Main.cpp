#include <omp.h>
#include <stdexcept>

#include "FilenamesParser/FilenamesParser.h"
#include "Main/Main.h"
#include "Tools/Logger.h"

namespace sbwt_search {

using log_utils::Logger;
using std::runtime_error;

Main::Main() { Logger::initialise_global_logging(Logger::LOG_LEVEL::WARN); }

auto Main::get_threads() const -> u64 { return threads; }

auto Main::load_threads() -> void {
  omp_set_nested(1);
#pragma omp parallel
#pragma omp single
  threads = omp_get_num_threads();
}

}  // namespace sbwt_search
