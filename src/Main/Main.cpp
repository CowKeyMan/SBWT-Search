#include <omp.h>
#include <stdexcept>

#include "FilenamesParser/FilenamesParser.h"
#include "Main/Main.h"
#include "Tools/Logger.h"

namespace sbwt_search {

using log_utils::Logger;
using std::runtime_error;

Main::Main() { Logger::initialise_global_logging(Logger::LOG_LEVEL::WARN); }

auto Main::get_input_filenames() const -> const vector<string> & {
  return input_filenames;
}
auto Main::get_output_filenames() const -> const vector<string> & {
  return output_filenames;
}
auto Main::get_threads() const -> u64 { return threads; }

auto Main::load_input_output_filenames(
  const string &input_file, const string &output_file
) -> void {
  FilenamesParser filenames_parser(input_file, output_file);
  input_filenames = filenames_parser.get_input_filenames();
  output_filenames = filenames_parser.get_output_filenames();
  if (input_filenames.size() != output_filenames.size()) {
    throw runtime_error("Input and output file sizes differ");
  }
}

auto Main::load_threads() -> void {
  omp_set_nested(1);
#pragma omp parallel
#pragma omp single
  threads = omp_get_num_threads();
}

}  // namespace sbwt_search
