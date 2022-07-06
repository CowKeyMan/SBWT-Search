#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "Builder/Builder.h"

using std::string;

namespace sbwt_search {

class BuilderDummy: public Builder {
  public:
    void check() { check_if_has_built(); }
};

TEST(BuilderTest, AlreadyBuilt) {
  auto host = BuilderDummy();
  host.check();
  try {
    host.check();
  } catch (std::logic_error &e) {
    ASSERT_EQ(string(e.what()), "Already Built");
  }
}

}
