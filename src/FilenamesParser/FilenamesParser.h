#ifndef FILENAMES_PARSER_H
#define FILENAMES_PARSER_H

/**
 * @file FilenamesParser.h
 * @brief Takes the input and output user input and outputs if the files are a
 * direct input or if they are a series of files. They are considered a series
 * of files if the extension of the file is '.list'
 */

#include <string>
#include <vector>

namespace sbwt_search {

using std::string;
using std::vector;

class FilenamesParser {
private:
  vector<string> input_filenames, output_filenames;

public:
  FilenamesParser(const string &input_filename, const string &output_filename);
  auto get_output_filenames() -> vector<string>;
  auto get_input_filenames() -> vector<string>;

private:
  auto is_list(const string &filename) -> bool;
  auto file_lines_to_vector(const string &filename) -> vector<string>;
};
}  // namespace sbwt_search

#endif
