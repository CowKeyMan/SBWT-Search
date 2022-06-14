#include <iostream>
#include <string>

#include "ArgumentParser.hpp"

using sbwt_search::parse_arguments;

auto main(int argc, char **argv) -> int { parse_arguments(argc, argv); }
