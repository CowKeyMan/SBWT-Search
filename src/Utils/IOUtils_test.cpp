#include <stdexcept>
#include <string>

#include "gtest/gtest.h"

#include "Utils/IOUtils.h"

using std::runtime_error;
using std::string;

namespace io_utils {

TEST(IOUtilsTest, TestFileExists) {
  ThrowingIfstream::check_file_exists("README.md");
  string random_file_name = "README.md_random_stuff_here";
  try {
    ThrowingIfstream::check_file_exists(random_file_name);
  } catch (runtime_error &e) {
    ASSERT_EQ(
      string(e.what()),
      "The input file " + random_file_name + " cannot be opened"
    );
  }
}

TEST(IOUtilsTest, TestPathValid) {
  ThrowingOfstream::check_path_valid("README.md");
  string random_path = "test_objects/tmpgarbage/test.txt";
  try {
    ThrowingOfstream::check_path_valid(random_path);
  } catch (runtime_error &e) {
    ASSERT_EQ(
      string(e.what()),
      string("The path ") + random_path
        + " cannot be opened. Check that all the folders in the path is "
          "correct and that you have permission to create files in this path "
          "folder"
    );
  }
}

}  // namespace io_utils
