#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "Parser.h"

using std::string;

namespace sbwt_search {

class ParserDummy: public Parser {
public:
  void check() { check_if_has_parsed(); }
};

TEST(ParserTest, AlreadyParsed) {
  auto host = ParserDummy();
  host.check();
  try {
    host.check();
  } catch (std::logic_error &e) {
    ASSERT_EQ(string(e.what()), "Already Parsed");
  }
}

}
