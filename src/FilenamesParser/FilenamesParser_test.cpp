#include <gtest/gtest.h>

#include "FilenamesParser/FilenamesParser.h"

namespace sbwt_search {

TEST(FilenamesParserTest, TestTxt) {
  auto host = FilenamesParser(
    "test_objects/filenames.list", "test_objects/filenames.list"
  );
  vector<string> expected_filenames = {"filename 1", "filename2", "filename_3"};
  EXPECT_EQ(host.get_input_filenames(), expected_filenames);
  EXPECT_EQ(host.get_output_filenames(), expected_filenames);
}

TEST(FilenamesParserTest, TestNotTxt) {
  auto host
    = FilenamesParser("test_objects/test_input.fna", "test_objects/out_file");
  vector<string> expected_input_files = {"test_objects/test_input.fna"};
  vector<string> expected_output_files = {"test_objects/out_file"};
  EXPECT_EQ(host.get_input_filenames(), expected_input_files);
  EXPECT_EQ(host.get_output_filenames(), expected_output_files);
}

}  // namespace sbwt_search
