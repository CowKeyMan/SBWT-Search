#include "gtest/gtest_pred_impl.h"

#include "FilenamesParser/FilenamesParser.h"
#include "Tools/TestUtils.hpp"

namespace sbwt_search {

using test_utils::assert_vectors_equal;

TEST(FilenamesParserTest, TestTxt) {
  auto host = FilenamesParser(
    "test_objects/filenames.list", "test_objects/filenames.list"
  );
  vector<vector<string>> expected_filenames
    = {{"filename 1"}, {"filename2", "filename_3"}};
  assert_vectors_equal(
    host.get_input_filenames(), expected_filenames, __FILE__, __LINE__
  );
  assert_vectors_equal(
    host.get_output_filenames(), expected_filenames, __FILE__, __LINE__
  );
}

TEST(FilenamesParserTest, TestNotTxt) {
  auto host
    = FilenamesParser("test_objects/test_input.fna", "test_objects/out_file");
  vector<vector<string>> expected_input_files
    = {{"test_objects/test_input.fna"}};
  vector<vector<string>> expected_output_files = {{"test_objects/out_file"}};
  assert_vectors_equal(
    host.get_input_filenames(), expected_input_files, __FILE__, __LINE__
  );
  assert_vectors_equal(
    host.get_output_filenames(), expected_output_files, __FILE__, __LINE__
  );
}

}  // namespace sbwt_search
