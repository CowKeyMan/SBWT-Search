#ifndef FILENAMES_PARSER_H
#define FILENAMES_PARSER_H

/**
 * @file FilenamesParser.h
 * @brief Takes the input and output user input and outputs if the files are a
 * direct input or if they are a series of files
 */

#include <string>
#include <vector>

using std::string;
using std::vector;

namespace sbwt_search {
class FilenamesParser {
private:
  vector<string> input_filenames, output_filenames;

public:
  FilenamesParser(string input_filename, string output_filename);
  vector<string> get_output_filenames();
  vector<string> get_input_filenames();

private:
  bool is_txt(string filename);
  vector<string> file_lines_to_vector(string filename);
};
}  // namespace sbwt_search

#endif
