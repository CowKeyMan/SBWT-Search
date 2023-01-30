#include "Main/IndexSearchMain.h"

using sbwt_search::IndexSearchMain;

auto main(int argc, char **argv) -> int {
  auto a = IndexSearchMain();
  a.main(argc, argv);
}
