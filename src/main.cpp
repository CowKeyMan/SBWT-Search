#include <algorithm>
#include <iostream>
#include <memory>
#include <span>
#include <string>

#include <unordered_map>

#include "Main/ColorSearchMain.h"
#include "Main/IndexSearchMain.h"
#include "Main/Main.h"

using sbwt_search::ColorSearchMain;
using sbwt_search::IndexSearchMain;
using sbwt_search::Main;
using std::cout;
using std::endl;
using std::make_shared;
using std::shared_ptr;
using std::span;
using std::string;
using std::unordered_map;

auto main(int argc, char **argv) -> int {
  auto args = span{argv, static_cast<u64>(argc)};
  const unordered_map<string, shared_ptr<Main>> str_to_item{
    {"index", make_shared<IndexSearchMain>()},
    {"colors", make_shared<ColorSearchMain>()}};
  if (argc == 1 || !str_to_item.contains(args[1])) {
    cout << "Usage: sbwt_search [index|colors]" << endl;
    return 1;
  }
  str_to_item.at(args[1])->main(argc, argv);
}
