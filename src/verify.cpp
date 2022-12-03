#include <iostream>
#include <memory>
#include <string>

#include "OutputParser/AsciiOutputParser.h"
#include "OutputParser/BinaryOutputParser.h"
#include "OutputParser/BoolOutputParser.h"
#include "OutputParser/OutputParser.h"
#include "cxxopts.hpp"

using cxxopts::Options;
using cxxopts::ParseResult;
using cxxopts::value;
using sbwt_search::AsciiOutputParser;
using sbwt_search::BinaryOutputParser;
using sbwt_search::BoolOutputParser;
using sbwt_search::ITEM_TYPE;
using sbwt_search::OutputParser;
using std::cerr;
using std::cout;
using std::endl;
using std::make_unique;
using std::runtime_error;
using std::string;
using std::unique_ptr;

auto get_cmd_arguments(int argc, char **argv) -> ParseResult;
auto get_output_parser(string filename, string format)
  -> unique_ptr<OutputParser>;
auto err_and_exit() -> void;
auto convert_value_to_bool(size_t value) -> bool;

size_t item_idx = 0, line_idx = 0;

auto main(int argc, char **argv) -> int {
  auto arguments = get_cmd_arguments(argc, argv);
  auto parser1 = get_output_parser(
    arguments["file1"].as<string>(), arguments["format1"].as<string>()
  );
  auto parser2 = get_output_parser(
    arguments["file2"].as<string>(), arguments["format2"].as<string>()
  );

  while (true) {
    auto item1 = parser1->get_next();
    auto item2 = parser2->get_next();
    if (item1 != item2) { err_and_exit(); }
    if (item1 == ITEM_TYPE::VALUE) {
      size_t value1 = parser1->get_value(), value2 = parser2->get_value();
      if (arguments["format1"].as<string>() == "bool") {
        value2 = convert_value_to_bool(value2);
      }
      if (arguments["format2"].as<string>() == "bool") {
        value1 = convert_value_to_bool(value1);
      }
      if (value1 != value2) { err_and_exit(); }
      ++item_idx;
    } else if (item1 == ITEM_TYPE::NEWLINE) {
      ++line_idx;
      item_idx = 0;
    } else if (item1 == ITEM_TYPE::EOF_T) {
      break;
    }
  }
  cout << "Files are equal. Success!" << endl;
}

auto err_and_exit() -> void {
  cerr << "Item mismatch at line " << line_idx << " item " << item_idx << endl;
  exit(1);
}

auto get_output_parser(string filename, string format)
  -> unique_ptr<OutputParser> {
  for (auto &c: format) { c = tolower(c); }
  if (format == "ascii") {
    return make_unique<AsciiOutputParser>(filename);
  } else if (format == "binary") {
    return make_unique<BinaryOutputParser>(filename);
  } else if (format == "bool") {
    return make_unique<BoolOutputParser>(filename);
  }
  throw runtime_error("Format " + format + " is not valid");
}

auto convert_value_to_bool(size_t value) -> bool {
  return !(value == size_t(-2) || value == size_t(-1));
}

auto get_cmd_arguments(int argc, char **argv) -> ParseResult {
  auto options = Options(
    "Verify", "Verify that results of ascii binary and bool are the same"
  );
  options.add_options()("i,file1", "First file", value<string>())(
    "o,file2", "Second file", value<string>()
  )("x,format1",
    "First file format of the first file, which may be 'ascii', 'binary' or "
    "'bool'. If if is bool "
    "format, it is also expected that there is another file with the name "
    "<input-file>_seq_sizes",
    value<string>()
  )("y,format2",
    "First file format of the second file, which may be 'ascii', 'binary' or "
    "'bool'. If if is bool "
    "format, it is also expected that there is another file with the name "
    "<input-file>_seq_sizes",
    value<string>()
  )("h,help", "Print usage", value<bool>()->default_value("false"));
  options.allow_unrecognised_options();
  auto arguments = options.parse(argc, argv);
  if (argc == 1 || arguments["help"].as<bool>()) {
    cout << options.help() << endl;
    exit(1);
  }
  return arguments;
}
