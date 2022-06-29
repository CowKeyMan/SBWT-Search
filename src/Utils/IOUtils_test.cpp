#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "IOUtils.hpp"

using std::string;
using std::runtime_error;

namespace sbwt_search {

TEST(IOUtilsTest, TestFileExists) {
  check_file_exists("README.md");
  string random_file_name = "README.md_random_stuff_here";
  try {
    check_file_exists(random_file_name.c_str());
  } catch (runtime_error &e) {
    ASSERT_EQ(string(e.what()), "The file " + random_file_name + " cannot be opened");
  }
}

}
