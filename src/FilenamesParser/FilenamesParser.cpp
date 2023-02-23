#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include "FilenamesParser/FilenamesParser.h"
#include "Tools/IOUtils.h"
#include "Tools/TypeDefinitions.h"

using io_utils::ThrowingIfstream;
using std::getline;
using std::ifstream;
using std::ios;
using std::string;
using std::vector;

namespace sbwt_search {

FilenamesParser::FilenamesParser(
  const string &input_filename, const string &output_filename
) {
  if (is_txt(input_filename)) {
    input_filenames = file_lines_to_vector(input_filename);
    output_filenames = file_lines_to_vector(output_filename);
  } else {
    input_filenames = {input_filename};
    output_filenames = {output_filename};
  }
}

auto FilenamesParser::is_txt(const string &filename) -> bool {
  u64 fsize = filename.size();
  return fsize >= 5 && filename.substr(fsize - 5, 5) == ".list";
}

auto FilenamesParser::file_lines_to_vector(const string &filename)
  -> vector<string> {
  vector<string> result;
  ThrowingIfstream stream(filename, ios::in);
  string buffer;
  while (getline(stream, buffer)) { result.push_back(buffer); }
  return result;
}

auto FilenamesParser::get_input_filenames() -> vector<string> {
  return input_filenames;
}
auto FilenamesParser::get_output_filenames() -> vector<string> {
  return output_filenames;
}

}  // namespace sbwt_search
