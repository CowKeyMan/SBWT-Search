#include <ios>
#include <string>

#include "gtest/gtest.h"

#include "Utils/IOUtils.h"
#include "fmt/core.h"

using fmt::format;

using std::ios;
using std::string;

namespace io_utils {

TEST(IOUtilsTest, TestFileExists) {
  ThrowingIfstream::check_file_exists("README.md");
  string random_file_name = "README.md_random_stuff_here";
  try {
    ThrowingIfstream::check_file_exists(random_file_name);
  } catch (ios::failure &e) {
    ASSERT_EQ(
      string(e.what()),
      format(
        "The input file {} cannot be opened: iostream error", random_file_name
      )
    );
  }
}

TEST(IOUtilsTest, TestPathValid) {
  ThrowingOfstream::check_path_valid("README.md");
  string random_path = "test_objects/tmpgarbage/test.txt";
  try {
    ThrowingOfstream::check_path_valid(random_path);
  } catch (ios::failure &e) {
    ASSERT_EQ(
      string(e.what()),
      format(
        "The path {} cannot be opened. Check that all the folders in the path "
        "is correct and that you have permission to create files in this path "
        "folder: iostream error",
        random_path
      )
    );
  }
}

}  // namespace io_utils
