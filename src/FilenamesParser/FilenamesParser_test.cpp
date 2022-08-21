#include "gtest/gtest_pred_impl.h"

#include "FilenamesParser/FilenamesParser.h"
#include "TestUtils/GeneralTestUtils.hpp"

namespace sbwt_search {

TEST(FilenamesParserTest, test_txt) {
  auto host = FilenamesParser(
    "test_objects/test_input.txt", "test_objects/test_output.txt"
  );
  vector<string> expected_input_files = { "a", "b", "c" };
  vector<string> expected_output_files
    = { "filename 1", "filename2", "filename_3" };
  assert_vectors_equal(host.get_input_filenames(), expected_input_files);
  assert_vectors_equal(host.get_output_filenames(), expected_output_files);
}

TEST(FilenamesParserTest, test_not_txt) {
  auto host = FilenamesParser(
    "test_objects/test_input.fna", "test_objects/test_output.txt"
  );
  vector<string> expected_input_files = { "test_objects/test_input.fna" };
  vector<string> expected_output_files = { "test_objects/test_output.txt" };
  assert_vectors_equal(host.get_input_filenames(), expected_input_files);
  assert_vectors_equal(host.get_output_filenames(), expected_output_files);
}

}  // namespace sbwt_search
